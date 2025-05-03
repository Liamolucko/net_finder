use std::iter;
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use anyhow::{bail, Context};
use indicatif::ProgressBar;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor, Instance, Limits,
    Maintain, MapMode, PipelineLayoutDescriptor, Queue, ShaderModuleDescriptor, ShaderSource,
    ShaderStages,
};

use net_finder::{
    Class, Cuboid, Cursor, Direction, Finder, FinderCtx, FinderInfo, Mapping, Runtime, Solution,
    SquareCache,
};
use Direction::*;

/// The number of `Finder`s we always give to the GPU.
const NUM_FINDERS: u32 = 13440;
/// The size of the compute shader's workgroups.
const WORKGROUP_SIZE: u32 = 64;
/// The capacity of the buffer into which our shader writes the solutions it
/// finds.
const SOLUTION_CAPACITY: u32 = 10000;

pub struct GpuRuntime<const CUBOIDS: usize> {
    cuboids: [Cuboid; CUBOIDS],
}

impl<const CUBOIDS: usize> GpuRuntime<CUBOIDS> {
    /// Makes a new runtime for running finders on the GPU.
    pub fn new(cuboids: [Cuboid; CUBOIDS]) -> Self {
        Self { cuboids }
    }
}

impl<const CUBOIDS: usize> Runtime<CUBOIDS> for GpuRuntime<CUBOIDS> {
    fn cuboids(&self) -> [Cuboid; CUBOIDS] {
        self.cuboids
    }

    fn run(
        self,
        ctx: &FinderCtx<CUBOIDS>,
        finders: &[FinderInfo<CUBOIDS>],
        solution_tx: &mpsc::Sender<Solution>,
        pause: &AtomicBool,
        progress: Option<&ProgressBar>,
    ) -> anyhow::Result<Vec<FinderInfo<CUBOIDS>>> {
        let mut finders = finders
            .iter()
            .map(|finder| Finder::new(ctx, finder))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut pipeline = pollster::block_on(Pipeline::new(ctx.clone(), &finders))?;

        while !finders.is_empty() && !pause.load(Ordering::Relaxed) {
            let (solutions, new_finders, finders_created, finders_finished) =
                pipeline.run_finders(mem::take(&mut finders));
            finders = new_finders;

            for solution in solutions {
                solution_tx.send(solution)?;
            }

            if let Some(progress) = progress {
                progress.inc_length(finders_created);
                progress.inc(finders_finished);
            }
        }

        Ok(finders
            .into_iter()
            .map(|finder| finder.into_info(ctx))
            .collect())
    }
}

/// Various useful constants that are often needed when talking to the GPU.
#[derive(Debug, Clone, Copy)]
struct Constants {
    /// The number of `u32`s needed to represent a list of decisions.
    decision_words: u32,
    /// How many bytes a `FinderInfo` takes up.
    finder_info_size: u32,
}

impl Constants {
    /// Computes a full set of `GpuConstants` from the fundamental two.
    fn compute(num_cuboids: u32, area: u32) -> Constants {
        let queue_capacity = 4 * area;

        let decision_words = queue_capacity.div_ceil(32);
        let finder_info_size = 4 * num_cuboids + 4 * decision_words + 8;

        Constants {
            decision_words,
            finder_info_size,
        }
    }
}

/// A GPU pipeline set up for running `Finder`s.
struct Pipeline<const CUBOIDS: usize> {
    device: Device,
    queue: Queue,

    /// The context that we need on the CPU to finish off the solutions the GPU
    /// gives us, split `Finder`s, etc.
    ctx: FinderCtx<CUBOIDS>,

    /// The surface area of the cuboids that this pipeline's `Finder`s search
    /// for.
    area: u32,

    pipeline: ComputePipeline,
    bind_group: BindGroup,

    finder_buf: Buffer,
    solution_buf: Buffer,
    cpu_finder_buf: Buffer,
    cpu_solution_buf: Buffer,
}

impl<const CUBOIDS: usize> Pipeline<CUBOIDS> {
    /// Creates a new GPU pipeline capable of running the given list of
    /// `Finder`s.
    ///
    /// The reason that the actual `Finder`s are needed rather than, say, a
    /// list of cuboids is to find out what the largest trie we need to allocate
    /// space for is.
    ///
    /// All the `Finder`s need to be searching for the common nets of the
    /// same list of cuboids.
    async fn new(ctx: FinderCtx<CUBOIDS>, finders: &[Finder<CUBOIDS>]) -> anyhow::Result<Self> {
        if finders.is_empty() {
            bail!("must specify at least one `Finder`");
        }

        let instance = Instance::new(&Default::default());
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
            .await
            .context("unable to create GPU adapter")?;
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    required_limits: Limits {
                        max_compute_workgroup_size_x: WORKGROUP_SIZE,
                        max_compute_invocations_per_workgroup: WORKGROUP_SIZE,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await?;

        // Compute all the constants that we have to prepend to the WGSL module.
        let num_cuboids: u32 = CUBOIDS.try_into()?;
        let area = ctx.target_area.try_into().unwrap();

        let Constants {
            finder_info_size, ..
        } = Constants::compute(num_cuboids, area);

        let mut neighbour_lookup_buf_contents = Vec::new();
        for cache in &ctx.square_caches {
            for square in cache.squares() {
                for direction in [Left, Up, Right, Down] {
                    neighbour_lookup_buf_contents.extend_from_slice(&u32::to_le_bytes(
                        Cursor::new(square, 0).moved_in(cache, direction).0.into(),
                    ));
                }
            }
        }
        let neighbour_lookup_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("neighbour lookup buffer"),
            contents: &neighbour_lookup_buf_contents,
            usage: BufferUsages::UNIFORM,
        });

        let mut class_lookup_buf_contents = Vec::new();
        for cache in &ctx.square_caches {
            for cursor in cache.cursors() {
                class_lookup_buf_contents
                    .extend_from_slice(&u32::to_le_bytes(cursor.class(cache).index().into()));
            }
        }
        let class_lookup_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("class lookup buffer"),
            contents: &class_lookup_buf_contents,
            usage: BufferUsages::UNIFORM,
        });

        let undo_lookup_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("undo lookup buffer"),
            contents: bytemuck::cast_slice(&undo_lookups(&ctx)),
            usage: BufferUsages::UNIFORM,
        });

        let finder_buf = device.create_buffer(&BufferDescriptor {
            label: Some("finder buffer"),
            size: u64::from(NUM_FINDERS * finder_info_size),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let solution_buf = device.create_buffer(&BufferDescriptor {
            label: Some("solution buffer"),
            size: u64::from(4 + SOLUTION_CAPACITY * finder_info_size),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Mapping buffers which are accessible by the GPU is apparently
        // inefficient, so we instead first copy the data out of our GPU buffers
        // into these buffers, then map and read from them instead.
        let cpu_finder_buf = device.create_buffer(&BufferDescriptor {
            label: Some("CPU finder buffer"),
            size: u64::from(NUM_FINDERS * finder_info_size),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let cpu_solution_buf = device.create_buffer(&BufferDescriptor {
            label: Some("CPU solution buffer"),
            size: u64::from(4 + SOLUTION_CAPACITY * finder_info_size),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("finder bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(neighbour_lookup_buf.size().try_into().unwrap()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(class_lookup_buf.size().try_into().unwrap()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(undo_lookup_buf.size().try_into().unwrap()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(finder_buf.size().try_into().unwrap()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(solution_buf.size().try_into().unwrap()),
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("finder bind group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &neighbour_lookup_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &class_lookup_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &undo_lookup_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &finder_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &solution_buf,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("finder pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader_source = include_str!("shader.wgsl").replace(
            "const num_cuboids: u32 = 2;\nconst area: u32 = 22;",
            &format!(
                "const num_cuboids: u32 = {};\nconst area: u32 = {};",
                CUBOIDS, ctx.target_area
            ),
        );

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("finder pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("run_finder"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,

            ctx,

            area,

            pipeline,
            bind_group,

            finder_buf,
            solution_buf,
            cpu_finder_buf,
            cpu_solution_buf,
        })
    }

    fn constants(&self) -> Constants {
        Constants::compute(CUBOIDS.try_into().unwrap(), self.area)
    }

    /// Runs the given finders for a bit and returns all the solutions they
    /// yield, as well as their state afterwards.
    ///
    /// Note that the number of finders returned may not be the same as the
    /// number passed in, since some finders may be split or finish.
    fn run_finders(
        &mut self,
        mut finders: Vec<Finder<CUBOIDS>>,
    ) -> (Vec<Solution>, Vec<Finder<CUBOIDS>>, u64, u64) {
        let mut solutions = Vec::new();

        let mut i = 0;
        let mut finders_created = 0;
        let mut finders_finished = 0;
        while finders.len() < NUM_FINDERS.try_into().unwrap() {
            if finders.is_empty() {
                // There's nothing for us to do.
                return (Vec::new(), Vec::new(), finders_created, finders_finished);
            }

            if i >= finders.len() {
                i -= finders.len();
            }
            if let Some(finder) = finders[i].split(&self.ctx) {
                finders.push(finder);
                i += 1;
                finders_created += 1;
            } else {
                // If we couldn't split it, it's finished.
                let finder = finders.remove(i);
                // However, it may be finished *with a solution*, so try and finalise this.
                solutions.extend(finder.finalize(&self.ctx));
                finders_finished += 1;
            }
        }

        self.device.start_capture();

        // Write the finders into the GPU's buffer.
        let mut gpu_finders = Vec::new();
        for finder in finders.into_iter() {
            finder.into_info(&self.ctx).to_gpu(self, &mut gpu_finders);
        }
        self.queue.write_buffer(&self.finder_buf, 0, &gpu_finders);

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        // Clear the `len` field of the solutions to 0.
        encoder.clear_buffer(&self.solution_buf, 0, Some(4));

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("finder pass"),
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(NUM_FINDERS / WORKGROUP_SIZE, 1, 1);
        }

        // After the GPU's done, we want to copy its results back into our CPU buffers.
        encoder.copy_buffer_to_buffer(
            &self.finder_buf,
            0,
            &self.cpu_finder_buf,
            0,
            self.finder_buf.size(),
        );
        encoder.copy_buffer_to_buffer(
            &self.solution_buf,
            0,
            &self.cpu_solution_buf,
            0,
            self.solution_buf.size(),
        );

        let index = self.queue.submit([encoder.finish()]);

        self.device.stop_capture();

        // Wait for the GPU to finish.
        self.device.poll(Maintain::WaitForSubmissionIndex(index));

        // Then we want to map those buffers.
        self.cpu_finder_buf
            .slice(..)
            .map_async(MapMode::Read, |result| result.unwrap());
        self.cpu_solution_buf
            .slice(..)
            .map_async(MapMode::Read, |result| result.unwrap());

        self.device.poll(Maintain::Wait);

        // I think our buffers are guaranteed to be mapped once `poll` returns? (not on
        // web but I'm not targeting that rn so I don't care)
        // So now extract our results.
        let raw_finders = self.cpu_finder_buf.slice(..).get_mapped_range();
        let raw_solutions = self.cpu_solution_buf.slice(..).get_mapped_range();

        let Constants {
            finder_info_size, ..
        } = self.constants();

        let num_solutions = u32::from_le_bytes(raw_solutions[0..4].try_into().unwrap());
        solutions.extend(
            raw_solutions[4..]
                .chunks(finder_info_size.try_into().unwrap())
                .take(num_solutions.try_into().unwrap())
                .flat_map(|bytes| {
                    let info = FinderInfo::from_gpu(self, bytes);
                    let mut finder = Finder::new(&self.ctx, &info).unwrap();
                    finder.finish_and_finalize(&self.ctx)
                }),
        );

        let mut finders: Vec<Finder<CUBOIDS>> = raw_finders
            .chunks(finder_info_size.try_into().unwrap())
            .map(|bytes| FinderInfo::from_gpu(self, bytes))
            .map(|info| Finder::new(&self.ctx, &info).unwrap())
            .collect();

        // Get rid of any finders that are finished.
        let old_len = finders.len();
        finders.retain_mut(|finder| {
            !(finder.index >= finder.queue.len()
                && finder.queue[finder.base_index..]
                    .iter()
                    .all(|instruction| instruction.followup_index.is_none()))
        });
        finders_finished += u64::try_from(old_len - finders.len()).unwrap();

        drop(raw_finders);
        drop(raw_solutions);
        self.cpu_finder_buf.unmap();
        self.cpu_solution_buf.unmap();

        (solutions, finders, finders_created, finders_finished)
    }
}

trait ToGpu<const CUBOIDS: usize> {
    fn to_gpu(&self, pipeline: &Pipeline<CUBOIDS>, buffer: &mut Vec<u8>);
}

trait FromGpu<const CUBOIDS: usize> {
    fn from_gpu(pipeline: &Pipeline<CUBOIDS>, bytes: &[u8]) -> Self;
}

impl<const CUBOIDS: usize> ToGpu<CUBOIDS> for FinderInfo<CUBOIDS> {
    fn to_gpu(&self, pipeline: &Pipeline<CUBOIDS>, buffer: &mut Vec<u8>) {
        let Constants { decision_words, .. } = pipeline.constants();
        // TODO: maybe we should pull out the equivalence classes and make sure this has
        // the minimal index
        //
        // although, I don't think it particularly matters anyway... it'll break the
        // fact that we compute fixed_class by just taking that class of
        // start_mapping[0], but that's fixable (mask out the transform bits).
        for cursor in pipeline
            .ctx
            .to_inner(self.start_mapping)
            .sample(&pipeline.ctx.square_caches)
            .cursors
        {
            buffer.extend_from_slice(&u32::to_le_bytes(cursor.0.into()));
        }
        for chunk in self
            .decisions
            .chunks(32)
            .chain(iter::repeat(&[] as &[bool]))
            .take(decision_words.try_into().unwrap())
        {
            let mut word = 0;
            for (i, &decision) in chunk.iter().enumerate() {
                word |= (decision as u32) << i;
            }
            buffer.extend_from_slice(&u32::to_le_bytes(word));
        }
        buffer.extend_from_slice(&u32::to_le_bytes(self.decisions.len().try_into().unwrap()));
        buffer.extend_from_slice(&u32::to_le_bytes(
            self.base_decision.get().try_into().unwrap(),
        ));
    }
}

impl<const CUBOIDS: usize> FromGpu<CUBOIDS> for FinderInfo<CUBOIDS> {
    fn from_gpu(pipeline: &Pipeline<CUBOIDS>, mut bytes: &[u8]) -> Self {
        let Constants { decision_words, .. } = pipeline.constants();

        let start_mapping = Mapping {
            cursors: bytes[..4 * CUBOIDS]
                .chunks(4)
                .map(|chunk| {
                    Cursor(
                        u32::from_le_bytes(chunk.try_into().unwrap())
                            .try_into()
                            .unwrap(),
                    )
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        };
        bytes = &bytes[4 * CUBOIDS..];

        let decisions: Vec<_> = bytes[..4 * usize::try_from(decision_words).unwrap()]
            .chunks(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        bytes = &bytes[4 * usize::try_from(decision_words).unwrap()..];

        let decisions_len = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let base_decision = u32::from_le_bytes(bytes[4..8].try_into().unwrap());

        FinderInfo {
            start_mapping: pipeline
                .ctx
                .to_outer(start_mapping.to_classes(&pipeline.ctx.square_caches)),
            decisions: (0..usize::try_from(decisions_len).unwrap())
                .map(|i| decisions[i / 32] & (1 << (i % 32)) != 0)
                .collect(),
            base_decision: usize::try_from(base_decision).unwrap().try_into().unwrap(),
        }
    }
}

/// Returns the contents of all the cuboids' undo lookups.
pub fn undo_lookups<const CUBOIDS: usize>(ctx: &FinderCtx<CUBOIDS>) -> Vec<u16> {
    ctx.square_caches
        .iter()
        .skip(1)
        .flat_map(|cache| undo_lookup(&ctx.square_caches[0], cache, ctx.fixed_class))
        .collect()
}

/// Returns the contents of a specific cuboid's undo lookup.
fn undo_lookup(
    fixed_square_cache: &SquareCache,
    cache: &SquareCache,
    fixed_class: Class,
) -> Vec<u16> {
    assert!(
        fixed_class.transform_bits() >= 2,
        "The GPU's skip checker assumes the fixed class has at least 2 transform bits (which is \
         true in practice until at least area 64)"
    );
    cache
        .cursors()
        .flat_map(|cursor| {
            (0..8).map(move |transform| {
                let class = cursor.class(cache);
                let options = fixed_class
                    .with_transform(transform)
                    .alternate_transforms(fixed_square_cache)
                    .map(|transform| class.undo_transform(cache, transform))
                    .collect::<Vec<_>>();

                assert!(options.len() <= 2);

                let hi = options[0].index() as u16;
                let lo = options.get(1).unwrap_or(&options[0]).index() as u16;

                (hi << 8) | lo
            })
        })
        .collect()
}
