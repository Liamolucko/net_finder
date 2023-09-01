use std::{
    array,
    collections::HashSet,
    iter, mem,
    num::NonZeroU64,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
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

use crate::{Cuboid, Cursor, Mapping, Net, NetFinder, NetPos, Solution, SquareCache, State};

use super::{finalize_inner, Instruction, InstructionState, PosState};

/// The number of `NetFinder`s we always give to the GPU.
const NUM_FINDERS: u32 = 1792;
/// The size of the compute shader's workgroups.
const WORKGROUP_SIZE: u32 = 64;
/// The capacity of the buffer into which our shader writes the solutions it
/// finds.
///
/// I don't think I've seen any sets of cuboids with 100000 or more solutions
/// yet so this should be a pretty safe upper bound.
const SOLUTION_CAPACITY: u32 = 10000;

/// A GPU pipeline set up for running `NetFinder`s.
struct Pipeline {
    device: Device,
    queue: Queue,

    /// The surface area of the cuboids that this pipeline's `NetFinder`s search
    /// for.
    area: u32,
    /// The maximum length of a `SkipSet` supported by this pipeline.
    trie_capacity: u32,

    pipeline: ComputePipeline,
    bind_group: BindGroup,

    finder_buf: Buffer,
    solution_buf: Buffer,
    cpu_finder_buf: Buffer,
    cpu_solution_buf: Buffer,
}

impl Pipeline {
    /// Creates a new GPU pipeline capable of running the given list of
    /// `NetFinder`s.
    ///
    /// The reason that the actual `NetFinder`s are needed rather than, say, a
    /// list of cuboids is to find out what the largest trie we need to allocate
    /// space for is.
    ///
    /// All the `NetFinder`s need to be searching for the common nets of the
    /// same list of cuboids.
    async fn new<const CUBOIDS: usize>(finders: &[NetFinder<CUBOIDS>]) -> anyhow::Result<Self> {
        if finders.is_empty() {
            bail!("must specify at least one `NetFinder`");
        }
        let cuboids = finders[0].cuboids;
        if finders[1..].iter().any(|finder| finder.cuboids != cuboids) {
            bail!("finders are operating on different lists of cuboids");
        }
        if cuboids[1..]
            .iter()
            .any(|cuboid| cuboid.surface_area() != cuboids[0].surface_area())
        {
            bail!("cuboids do not all have the same surface area");
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
                    limits: Limits::default(),
                },
                None,
            )
            .await?;

        // Compute all the constants that we have to prepend to the WGSL module.
        let num_cuboids: u32 = CUBOIDS.try_into()?;
        let area = cuboids[0].surface_area().try_into()?;
        let trie_capacity = finders
            .iter()
            .map(|finder| finder.skip.data.len())
            .max()
            .unwrap()
            .try_into()?;

        let net_width = 2 * area + 1;
        let net_len = net_width * net_width;

        let num_squares = area;
        let num_cursors = 4 * num_squares;
        let queue_capacity = num_cursors;

        let trie_start = num_cuboids;

        let surface_words = (num_squares + 31) / 32;

        let instruction_size = 4 * (2 + num_cuboids);
        let queue_size = instruction_size * queue_capacity + 4;
        let potential_size = 4 * (queue_capacity + 1);
        let pos_state_size = 4 * (3 + num_cuboids);
        let finder_size = 4 * trie_capacity
            + queue_size
            + potential_size
            + 8
            + pos_state_size * net_len
            + 4 * surface_words * num_cuboids;

        let solution_size = 2 * queue_size;

        let mut lookup_buf_contents = Vec::new();
        for cache in &finders[0].square_caches {
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
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
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
                        min_binding_size: Some(cpu_finder_buf.size().try_into().unwrap()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(cpu_solution_buf.size().try_into().unwrap()),
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
            const trie_capacity = {trie_capacity}u;

            const net_width = {net_width}u;
            const net_len = {net_len}u;

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

            area,
            trie_capacity,

            pipeline,
            bind_group,

            finder_buf,
            solution_buf,
            cpu_finder_buf,
            cpu_solution_buf,
        })
    }

    /// Runs the given finders for a bit and returns all the solutions they
    /// yield, as well as their state afterwards.
    ///
    /// Note that the number of finders returned may not be the same as the
    /// number passed in, since some finders may be split or finish.
    fn run_finders<const CUBOIDS: usize>(
        &mut self,
        mut finders: Vec<NetFinder<CUBOIDS>>,
        prior_search_time: Duration,
        start: Instant,
    ) -> (Vec<Solution>, Vec<NetFinder<CUBOIDS>>) {
        let mut solutions = Vec::new();

        let mut i = 0;
        while finders.len() < NUM_FINDERS.try_into().unwrap() {
            if finders.is_empty() {
                // There's nothing for us to do.
                return (Vec::new(), Vec::new());
            }

            if i >= finders.len() {
                i -= finders.len();
            }
            if let Some(finder) = finders[i].split() {
                finders.push(finder);
                i += 1;
            } else {
                // If we couldn't split it, it's finished.
                let finder = finders.remove(i);
                // However, it may be finished *with a solution*, so try and finalise this.
                solutions.extend(finder.finalize(prior_search_time, start));
            }
        }

        // sike we're just gonna run the first one on the CPU now for a bit
        // let finder = &mut finders[0];
        // let mut i = 1;
        // loop {
        //     println!("{i}: {}", finder.queue.len());
        //     if i == 23 {
        //         for (i, instruction) in finder.queue.iter().enumerate() {
        //             println!(
        //                 "{i}: {} -> {:?}",
        //                 instruction.net_pos.0,
        //                 instruction.mapping.cursors.map(|cursor| cursor.0)
        //             );
        //         }
        //     }
        //     if finder.index < finder.queue.len() {
        //         println!("{:?}", finder.queue[finder.index]);
        //         finder.handle_instruction()
        //     } else {
        //         if !finder.backtrack() {
        //             break;
        //         }
        //     }
        //     i += 1;
        // }

        // panic!();

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
        encoder.clear_buffer(&self.cpu_solution_buf, 0, Some(NonZeroU64::new(4).unwrap()));

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
        let _raw_finders = self.cpu_finder_buf.slice(..).get_mapped_range();
        let raw_solutions = self.cpu_solution_buf.slice(..).get_mapped_range();

        let num_solutions = u32::from_le_bytes(raw_solutions[0..4].try_into().unwrap());
        let solutions = (0..num_solutions.try_into().unwrap())
            .flat_map(|i| {
                let queue_capacity = usize::try_from(4 * self.area).unwrap();
                let instruction_size = 4 * (2 + CUBOIDS);
                let completed_offset = 4 + 2 * (queue_capacity * instruction_size + 4) * i;
                let completed_len_offset = completed_offset + instruction_size * queue_capacity;
                let potential_offset = completed_len_offset + 4;
                let potential_len_offset = potential_offset + instruction_size * queue_capacity;

                let completed_len = u32::from_le_bytes(
                    raw_solutions[completed_len_offset..completed_len_offset + 4]
                        .try_into()
                        .unwrap(),
                );
                let potential_len = u32::from_le_bytes(
                    raw_solutions[potential_len_offset..potential_len_offset + 4]
                        .try_into()
                        .unwrap(),
                );

                let completed: Vec<_> = (0..usize::try_from(completed_len).unwrap())
                    .map(|i| {
                        let offset = completed_offset + i * instruction_size;
                        Instruction::from_gpu(&raw_solutions[offset..offset + instruction_size])
                    })
                    .collect();
                let potential: Vec<_> = (0..usize::try_from(potential_len).unwrap())
                    .map(|i| {
                        let offset = potential_offset + i * instruction_size;
                        Instruction::from_gpu(&raw_solutions[offset..offset + instruction_size])
                    })
                    .collect();

                finalize_inner(
                    (2 * self.area + 1).try_into().unwrap(),
                    &finders[0].square_caches,
                    &completed,
                    potential,
                    prior_search_time,
                    start,
                )
            })
            .collect();

        // TODO: finder deserialisation
        (solutions, Vec::new())
    }
}

impl SquareCache {
    fn to_gpu(&self, buffer: &mut Vec<u8>) {
        for cursor in &self.neighbour_lookup {
            buffer.extend_from_slice(&u32::to_le_bytes(cursor.0.into()));
        }
    }
}

impl<const CUBOIDS: usize> NetFinder<CUBOIDS> {
    fn to_gpu(&self, pipeline: &Pipeline, buffer: &mut Vec<u8>) {
        let trie_capacity = pipeline.trie_capacity;
        let queue_capacity = 4 * pipeline.area;

        // Write `skip`.
        buffer.extend(
            self.skip
                .data
                .iter()
                .copied()
                .chain(iter::repeat(0))
                .take(trie_capacity.try_into().unwrap())
                .flat_map(u32::to_le_bytes),
        );

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

        // Write `pos_states`.
        for row in self.pos_states.rows() {
            for pos_state in row {
                pos_state.to_gpu(buffer);
            }
        }

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
    }
}

impl<const CUBOIDS: usize> Instruction<CUBOIDS> {
    fn to_gpu(&self, buffer: &mut Vec<u8>) {
        buffer.extend_from_slice(&u32::to_le_bytes(self.net_pos.0.try_into().unwrap()));
        buffer.extend(
            self.mapping
                .cursors()
                .iter()
                .flat_map(|&cursor| u32::to_le_bytes(cursor.0.into())),
        );
        buffer.extend_from_slice(&u32::to_le_bytes(match self.state {
            InstructionState::NotRun => 0,
            InstructionState::Completed { followup_index } => followup_index.try_into().unwrap(),
        }))
    }

    fn from_gpu(bytes: &[u8]) -> Self {
        let net_pos = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
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
            net_pos: NetPos(net_pos.try_into().unwrap()),
            mapping,
            state: match followup_index {
                0 => InstructionState::NotRun,
                _ => InstructionState::Completed {
                    followup_index: followup_index.try_into().unwrap(),
                },
            },
        }
    }
}

impl<const CUBOIDS: usize> PosState<CUBOIDS> {
    fn to_gpu(&self, buffer: &mut Vec<u8>) {
        let (state, mapping, setter, conflict) = match *self {
            PosState::Unknown => (0, [Cursor(0); CUBOIDS], 0, 0),
            PosState::Known { mapping, setter } => (1, mapping.cursors, setter, 0),
            PosState::Invalid {
                mapping,
                setter,
                conflict,
            } => (2, mapping.cursors, setter, conflict),
            PosState::Filled { mapping, setter } => (3, mapping.cursors, setter, 0),
        };
        buffer.extend_from_slice(&u32::to_le_bytes(state));
        for Cursor(index) in mapping {
            buffer.extend_from_slice(&u32::to_le_bytes(index.into()));
        }
        buffer.extend_from_slice(&u32::to_le_bytes(setter.try_into().unwrap()));
        buffer.extend_from_slice(&u32::to_le_bytes(conflict.try_into().unwrap()));
    }
}

pub fn run<const CUBOIDS: usize>(
    _cuboids: [Cuboid; CUBOIDS],
    state: Arc<Mutex<State<CUBOIDS>>>,
    mut yielded_nets: HashSet<Net>,
    _progress: ProgressBar,
) -> impl Iterator<Item = Solution> {
    let guard = state.lock().unwrap();
    let mut finders = guard.finders.clone();
    let prior_search_time = guard.prior_search_time;
    drop(guard);
    let start = Instant::now();

    let mut pipeline = pollster::block_on(Pipeline::new(&finders)).unwrap();
    let mut new_solutions = Vec::new();

    std::iter::from_fn(move || {
        while !finders.is_empty() && new_solutions.is_empty() {
            (new_solutions, finders) =
                pipeline.run_finders(mem::take(&mut finders), prior_search_time, start);
            new_solutions.retain(|solution| {
                let new = yielded_nets.insert(solution.net.clone());
                new
            });
            let mut state = state.lock().unwrap();
            state.finders = finders.clone();
            state.solutions.extend(new_solutions.iter().cloned());
            state.prior_search_time = prior_search_time + start.elapsed();
        }
        if let Some(solution) = new_solutions.pop() {
            Some(solution)
        } else {
            // We must be out of finders if we broke out of that loop, so we're done.
            None
        }
    })
}
