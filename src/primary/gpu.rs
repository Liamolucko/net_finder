use std::{
    array,
    collections::HashSet,
    iter, mem,
    num::{NonZeroU64, NonZeroU8},
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{bail, Context};
use indicatif::ProgressBar;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor, Dx12Compiler, Features,
    Instance, InstanceDescriptor, Limits, Maintain, MapMode, PipelineLayoutDescriptor, Queue,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::{Cursor, Finder, Mapping, Net, Pos, Solution, SquareCache, State, Surface};

use super::{set::InstructionSet, FinderCtx, Instruction};

/// The number of `Finder`s we always give to the GPU.
const NUM_FINDERS: u32 = 13440;
/// The size of the compute shader's workgroups.
const WORKGROUP_SIZE: u32 = 64;
/// The capacity of the buffer into which our shader writes the solutions it
/// finds.
const SOLUTION_CAPACITY: u32 = 10000;

/// Various useful constants that are often needed when talking to the GPU.
///
/// The first batch of them are the constants that we actually prepend to the
/// GPU shader; then there's the sizes in bytes of various things.
#[derive(Debug, Clone, Copy)]
// even if some fields are unused, I still want to keep them around since they're needed to
// calculate the rest.
#[allow(unused)]
struct Constants {
    // The three fundamental numbers from which everything else is computed.
    /// The number of cuboids we're finding common nets for.
    num_cuboids: u32,
    /// The shared area of those cuboids.
    area: u32,

    /// The number of squares on the surfaces of the cuboids (`area`).
    num_squares: u32,
    /// The number of cursors on the surfaces of the cuboids (`4 *
    /// num_squares`).
    num_cursors: u32,
    /// The capacity of `Finder`'s queue (`4 * num_squares`).
    ///
    /// This is `4 * num_squares` because there can be at most 4 instructions
    /// trying to set each square, one coming from each direction.
    queue_capacity: u32,

    /// The index of the start of the page where searching should start in the
    /// trie (`num_cursors`).
    ///
    /// This is `num_cursors` because that's the length of each normal page in
    /// the trie.
    trie_start: u32,

    /// The number of `u32`s needed to represent the squares filled on the
    /// surface (`num_squares.div_ceil(32)`).
    surface_words: u32,
    /// How many bytes an `Instruction` takes up (`4 * (2 + num_cuboids)`).
    instruction_size: u32,

    /// How many bytes a `Queue` takes up (`queue_capacity * instruction_size +
    /// 4`).
    queue_size: u32,
    /// How many bytes `Finder::potential` + `Finder::potential_len` takes up
    /// (`4 * (queue_capacity + 1)`).
    potential_size: u32,
    /// How many bytes `Finder::surfaces` takes up (`4 * surface_words *
    /// num_cuboids`).
    surfaces_size: u32,
    /// How many bytes of padding there are before `Finder::queued`.
    queued_padding: u32,
    /// How many bytes `Finder::queued` takes up (`4 * 4 * num_squares`).
    queued_size: u32,

    /// How many bytes a `Finder` takes up (`trie_size + queue_size +
    /// potential_size + 8 + surfaces_size + queued_padding + queued_size`).
    finder_size: u32,
    /// The size of a `MaybeSolution` (`2 * queue_size`).
    solution_size: u32,
}

impl Constants {
    /// Computes a full set of `GpuConstants` from the fundamental three.
    fn compute(num_cuboids: u32, area: u32) -> Constants {
        let num_squares = area;
        let num_cursors = 4 * num_squares;
        let queue_capacity = 4 * num_squares;

        let trie_start = num_cursors;

        let surface_words = (num_squares + 31) / 32;

        let instruction_size = 4 * (2 + num_cuboids);

        let queue_size = queue_capacity * instruction_size + 4;
        let potential_size = 4 * (queue_capacity + 1);
        let surfaces_size = 4 * surface_words * num_cuboids;
        let queued_size = 4 * 4 * num_squares;

        let unpadded_finder_size = queue_size + potential_size + 8 + surfaces_size + queued_size;
        let queued_padding = match unpadded_finder_size % 16 {
            0 => 0,
            x => 16 - x,
        };
        let finder_size = unpadded_finder_size + queued_padding;
        let solution_size = 2 * queue_size;

        Constants {
            num_cuboids,
            area,
            num_squares,
            num_cursors,
            queue_capacity,
            trie_start,
            surface_words,
            instruction_size,
            queue_size,
            potential_size,
            surfaces_size,
            queued_padding,
            queued_size,
            finder_size,
            solution_size,
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

        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::PRIMARY,
            dx12_shader_compiler: Dx12Compiler::Fxc,
        });
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
            .await
            .context("unable to create GPU adapter")?;
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    features: Features::empty(),
                    limits: Limits {
                        max_compute_workgroup_size_x: WORKGROUP_SIZE,
                        max_compute_invocations_per_workgroup: WORKGROUP_SIZE,
                        ..Limits::default()
                    },
                },
                None,
            )
            .await?;

        // Compute all the constants that we have to prepend to the WGSL module.
        let num_cuboids: u32 = CUBOIDS.try_into()?;
        let area = ctx.target_area.try_into().unwrap();

        let Constants {
            num_squares,
            num_cursors,
            queue_capacity,
            trie_start,
            surface_words,
            finder_size,
            solution_size,
            ..
        } = Constants::compute(num_cuboids, area);

        let mut lookup_buf_contents = Vec::new();
        for cache in &ctx.square_caches {
            cache.to_gpu(&mut lookup_buf_contents);
        }
        let lookup_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("neighbour lookup buffer"),
            contents: &lookup_buf_contents,
            usage: BufferUsages::UNIFORM,
        });

        let finder_buf = device.create_buffer(&BufferDescriptor {
            label: Some("finder buffer"),
            size: u64::from(NUM_FINDERS * finder_size),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let solution_buf = device.create_buffer(&BufferDescriptor {
            label: Some("solution buffer"),
            size: u64::from(4 + SOLUTION_CAPACITY * solution_size),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Mapping buffers which are accessible by the GPU is apparently
        // inefficient, so we instead first copy the data out of our GPU buffers
        // into these buffers, then map and read from them instead.
        let cpu_finder_buf = device.create_buffer(&BufferDescriptor {
            label: Some("CPU finder buffer"),
            size: u64::from(NUM_FINDERS * finder_size),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let cpu_solution_buf = device.create_buffer(&BufferDescriptor {
            label: Some("CPU solution buffer"),
            size: u64::from(4 + SOLUTION_CAPACITY * solution_size),
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
                        min_binding_size: Some(lookup_buf.size().try_into().unwrap()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(finder_buf.size().try_into().unwrap()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
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
                        buffer: &lookup_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &finder_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 2,
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

        let shader_source = format!(
            "\
            const num_cuboids = {num_cuboids}u;
            const area = {area}u;

            const num_squares = {num_squares}u;
            const num_cursors = {num_cursors}u;
            const queue_capacity = {queue_capacity}u;

            const trie_start = {trie_start}u;

            const surface_words = {surface_words}u;

            {}",
            include_str!("shader.wgsl")
        );

        // Bounds checks in Metal are currently broken (https://github.com/gfx-rs/naga/issues/2437),
        // so don't include them.
        let shader = unsafe {
            device.create_shader_module_unchecked(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(shader_source.into()),
            })
        };

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("finder pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "run_finder",
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
        for finder in finders.iter() {
            finder.to_gpu(self, &mut gpu_finders);
        }
        self.queue.write_buffer(&self.finder_buf, 0, &gpu_finders);

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        // Clear the `len` field of the solutions to 0.
        encoder.clear_buffer(&self.solution_buf, 0, Some(NonZeroU64::new(4).unwrap()));

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("finder pass"),
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
            instruction_size,
            queue_size,
            finder_size,
            solution_size,
            ..
        } = self.constants();

        let num_solutions = u32::from_le_bytes(raw_solutions[0..4].try_into().unwrap());
        solutions.extend(
            raw_solutions[4..]
                .chunks(solution_size.try_into().unwrap())
                .take(num_solutions.try_into().unwrap())
                .flat_map(|bytes| {
                    let queue_size: usize = queue_size.try_into().unwrap();

                    let completed_offset = 0;
                    let potential_offset = queue_size;
                    let end_offset = 2 * queue_size;

                    let completed_len = u32::from_le_bytes(
                        bytes[potential_offset - 4..potential_offset]
                            .try_into()
                            .unwrap(),
                    );
                    let potential_len =
                        u32::from_le_bytes(bytes[end_offset - 4..end_offset].try_into().unwrap());

                    let completed: Vec<_> = bytes[completed_offset..potential_offset - 4]
                        .chunks(instruction_size.try_into().unwrap())
                        .take(completed_len.try_into().unwrap())
                        .map(Instruction::from_gpu)
                        .collect();
                    let potential: Vec<_> = bytes[potential_offset..end_offset - 4]
                        .chunks(instruction_size.try_into().unwrap())
                        .take(potential_len.try_into().unwrap())
                        .map(Instruction::from_gpu)
                        .collect();

                    self.ctx.finalize(completed, potential)
                }),
        );

        let mut finders: Vec<Finder<CUBOIDS>> = raw_finders
            .chunks(finder_size.try_into().unwrap())
            .map(|bytes| Finder::from_gpu(self, bytes))
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

impl SquareCache {
    fn to_gpu(&self, buffer: &mut Vec<u8>) {
        for cursor in &self.neighbour_lookup {
            buffer.extend_from_slice(&u32::to_le_bytes(cursor.0.into()));
        }
    }
}

impl<const CUBOIDS: usize> Finder<CUBOIDS> {
    fn to_gpu(&self, pipeline: &Pipeline<CUBOIDS>, buffer: &mut Vec<u8>) {
        let Constants {
            queue_capacity,
            queued_padding,
            ..
        } = pipeline.constants();

        // Write `queue`.
        for i in 0..queue_capacity {
            if let Some(instruction) = self.queue.get(usize::try_from(i).unwrap()) {
                instruction.to_gpu(buffer);
            } else {
                buffer.extend(iter::repeat(0).take(4 * (2 + CUBOIDS)));
            }
        }
        buffer.extend_from_slice(&u32::to_le_bytes(self.queue.len().try_into().unwrap()));

        // Write `potential` and `potential_len`.
        for i in 0..queue_capacity {
            buffer.extend_from_slice(&u32::to_le_bytes(
                if let Some(&index) = self.potential.get(usize::try_from(i).unwrap()) {
                    index.try_into().unwrap()
                } else {
                    0
                },
            ));
        }
        buffer.extend_from_slice(&u32::to_le_bytes(self.potential.len().try_into().unwrap()));

        // Write `index` and `base_index`.
        buffer.extend_from_slice(&u32::to_le_bytes(self.index.try_into().unwrap()));
        buffer.extend_from_slice(&u32::to_le_bytes(self.base_index.try_into().unwrap()));

        // Write `surfaces`.
        for surface in &self.surfaces {
            if pipeline.area > 32 {
                // Since the first 32 squares correspond to the first `u32` on the GPU and the
                // lower 32 bits of the surface on the CPU, just encoding the whole integer as
                // little-endian happens to lay it out correctly.
                buffer.extend_from_slice(&surface.0.to_le_bytes());
            } else {
                // Otherwise, since we pack everything into the lowest `area` bits, this should
                // be able to fit into a `u32`.
                buffer.extend_from_slice(&u32::to_le_bytes(surface.0.try_into().unwrap()));
            }
        }

        // Write `queued`.
        buffer.extend(iter::repeat(0).take(queued_padding.try_into().unwrap()));
        for &(length, positions) in &self.queued.elements {
            let mut positions = bytemuck::cast::<u64, [Pos; 4]>(positions);
            if cfg!(target_endian = "big") {
                positions.reverse();
            }
            let encoded_positions: [u32; 4] = array::from_fn(|i| {
                let mut result = 0;
                if i == 0 {
                    result |= length << 28;
                }
                if i < length.try_into().unwrap() {
                    result |= 0x80000000;
                    result |= (positions[i].x as u32) << 8;
                    result |= positions[i].y as u32;
                }
                result
            });
            for word in encoded_positions {
                buffer.extend_from_slice(&u32::to_le_bytes(word));
            }
        }
    }

    fn from_gpu(pipeline: &Pipeline<CUBOIDS>, mut bytes: &[u8]) -> Self {
        let Constants {
            num_cursors,
            surface_words,
            instruction_size,
            queue_size,
            potential_size,
            surfaces_size,
            queued_padding,
            queued_size,
            ..
        } = pipeline.constants();

        let queue_size: usize = queue_size.try_into().unwrap();
        let queue_len = u32::from_le_bytes(bytes[queue_size - 4..queue_size].try_into().unwrap());
        let queue = bytes[0..queue_size - 4]
            .chunks(instruction_size.try_into().unwrap())
            .take(queue_len.try_into().unwrap())
            .map(Instruction::from_gpu)
            .collect();
        bytes = &bytes[queue_size..];

        let potential_size: usize = potential_size.try_into().unwrap();
        let potential_len = u32::from_le_bytes(
            bytes[potential_size - 4..potential_size]
                .try_into()
                .unwrap(),
        );
        let potential: Vec<usize> = bytemuck::cast_slice(&bytes[..potential_size])
            [..potential_len.try_into().unwrap()]
            .iter()
            .map(|&index| u32::from_le(index).try_into().unwrap())
            .collect();
        bytes = &bytes[potential_size..];

        let index = u32::from_le_bytes(bytes[0..4].try_into().unwrap())
            .try_into()
            .unwrap();
        let base_index = u32::from_le_bytes(bytes[4..8].try_into().unwrap())
            .try_into()
            .unwrap();
        bytes = &bytes[8..];

        let surfaces_size: usize = surfaces_size.try_into().unwrap();
        let surfaces = match surface_words {
            1 => bytemuck::from_bytes::<[u32; CUBOIDS]>(&bytes[..surfaces_size])
                .map(u32::from_le)
                .map(u64::from),
            2 => bytemuck::from_bytes::<[u64; CUBOIDS]>(&bytes[..surfaces_size]).map(u64::from_le),
            _ => unreachable!("area > 64"),
        }
        .map(Surface);
        bytes = &bytes[surfaces_size..];

        bytes = &bytes[queued_padding.try_into().unwrap()..];
        let queued_size: usize = queued_size.try_into().unwrap();
        let elements = bytes[..queued_size]
            .chunks(4 * 4)
            .map(|bytes| {
                let positions = bytemuck::from_bytes::<[u32; 4]>(bytes).map(u32::from_le);
                let length = (positions[0] >> 28) & 0x7;
                let mut positions = positions.map(|pos| Pos {
                    x: (pos >> 8) as u8,
                    y: pos as u8,
                });
                if cfg!(target_endian = "big") {
                    positions.reverse();
                }
                let positions: u64 = bytemuck::cast(positions);
                (length, positions)
            })
            .collect();
        let queued = InstructionSet { elements };

        Self {
            start_mapping_index: todo!(),
            queue,
            potential,
            queued,
            surfaces,
            index,
            base_index,
        }
    }
}

impl<const CUBOIDS: usize> Instruction<CUBOIDS> {
    fn to_gpu(&self, buffer: &mut Vec<u8>) {
        buffer.extend_from_slice(&u32::to_le_bytes(
            ((self.net_pos.x as u32) << 8) | (self.net_pos.y as u32),
        ));
        buffer.extend(
            self.mapping
                .cursors()
                .iter()
                .flat_map(|&cursor| u32::to_le_bytes(cursor.0.into())),
        );
        buffer.extend_from_slice(&u32::to_le_bytes(
            self.followup_index.map_or(0, NonZeroU8::get).into(),
        ))
    }

    fn from_gpu(bytes: &[u8]) -> Self {
        let encoded_net_pos = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let net_pos = Pos {
            x: (encoded_net_pos >> 8) as u8,
            y: encoded_net_pos as u8,
        };
        let mapping = Mapping {
            cursors: array::from_fn(|i| {
                let offset = 4 + 4 * i;
                let index = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                Cursor(index.try_into().unwrap())
            }),
        };
        let offset = 4 + 4 * CUBOIDS;
        let followup_index = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        Self {
            net_pos,
            mapping,
            // this is just an optimisation, we can leave it unset for now.
            skip: None,
            followup_index: NonZeroU8::new(followup_index.try_into().unwrap()),
        }
    }
}

pub fn run<const CUBOIDS: usize>(
    state: Arc<Mutex<State<CUBOIDS>>>,
    mut yielded_nets: HashSet<Net>,
    progress: ProgressBar,
) -> anyhow::Result<impl Iterator<Item = Solution>> {
    let guard = state.lock().unwrap();
    let cuboids = guard.cuboids;
    let prior_search_time = guard.prior_search_time;
    let start = Instant::now();

    let mut finders = guard.finders.clone();
    drop(guard);

    let ctx = FinderCtx::new(cuboids, prior_search_time)?;

    let mut pipeline = pollster::block_on(Pipeline::new(ctx, &finders)).unwrap();
    // The list of solutions we're partway through yielding.
    let mut yielding = Vec::new();

    Ok(std::iter::from_fn(move || {
        while !finders.is_empty() && yielding.is_empty() {
            let (mut solutions, new_finders, finders_created, finders_finished) =
                pipeline.run_finders(mem::take(&mut finders));
            finders = new_finders;

            solutions.retain(|solution| {
                let new = yielded_nets.insert(solution.net.clone());
                new
            });
            yielding = solutions;

            let mut state = state.lock().unwrap();
            state.finders = finders.clone();
            state.solutions.extend(yielding.iter().cloned());
            state.prior_search_time = prior_search_time + start.elapsed();

            progress.inc_length(finders_created);
            progress.inc(finders_finished);
        }
        if let Some(solution) = yielding.pop() {
            Some(solution)
        } else {
            // We must be out of finders if we broke out of that loop, so we're done.
            None
        }
    }))
}
