// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This example demonstrates how to use the compute capabilities of Vulkan.
//
// While graphics cards have traditionally been used for graphical operations, over time they have
// been more or more used for general-purpose operations as well. This is called "General-Purpose
// GPU", or *GPGPU*. This is what this example demonstrates.

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync;
use vulkano::sync::GpuFuture;

use std::sync::Arc;

fn main() {
    // As with other examples, the first step is to create an instance.
    let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();

    // Choose which physical device to use.
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    // Choose the queue of the physical device which is going to run our compute operation.
    //
    // The Vulkan specs guarantee that a compliant implementation must provide at least one queue
    // that supports compute operations.
    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_compute())
        .unwrap();

    // Now initializing the device.
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::none()
        },
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
    // example we use only one queue, so we just retrieve the first and only element of the
    // iterator and throw it away.
    let queue = queues.next().unwrap();

    println!("Device initialized");

    // Now let's get to the actual example.
    //
    // What we are going to do is very basic: we are going to fill a buffer with 64k integers
    // and ask the GPU to multiply each of them by 12.
    //
    // GPUs are very good at parallel computations (SIMD-like operations), and thus will do this
    // much more quickly than a CPU would do. While a CPU would typically multiply them one by one
    // or four by four, a GPU will do it by groups of 32 or 64.
    //
    // Note however that in a real-life situation for such a simple operation the cost of
    // accessing memory usually outweighs the benefits of a faster calculation. Since both the CPU
    // and the GPU will need to access data, there is no other choice but to transfer the data
    // through the slow PCI express bus.

    // We need to create the compute pipeline that describes our operation.
    //
    // If you are familiar with graphics pipeline, the principle is the same except that compute
    // pipelines are much simpler to create.
    let (pipeline, push_constants) = {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                define: [("TILE_SIZE", "16"),
                         ("DTYPE", "uint")],
                src: "
                    #version 450

                    layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;

                    layout(set = 0, binding = 0) buffer A {
                        DTYPE data[];
                    } a;

                    layout(set = 0, binding = 1) buffer B {
                        DTYPE data[];
                    } b;

                    layout(set = 0, binding = 2) buffer C {
                        DTYPE data[];
                    } c;

                    layout(push_constant) uniform PushConstantData {
                        uint shared_dim;
                        uint ret_cols;
                        uint ret_rows;
                      } pc;
                    // shared memory stores tiles from A and B per group
                    // threads in the same group will be able to use data other threads have writting to shared memory
                    // without having to step up caches and potentially missing
                    // this is the power of the tiling method
                    // data locality
                    shared DTYPE tile_A[TILE_SIZE][TILE_SIZE];
                    shared DTYPE tile_B[TILE_SIZE][TILE_SIZE];
                    
                    void main() {

                        uint thread_col = gl_LocalInvocationID.x; 
                        uint thread_row = gl_LocalInvocationID.y;
                        uint c_row = gl_GlobalInvocationID.y;
                        uint c_col = gl_GlobalInvocationID.x;
                        DTYPE acc = 0;
                        // tiled along the shared dimension
                        // we will iterate shared_dim/TILE_SIZE times i.e. the numer of tiles it takes to calculate a tile's worth of elements in C
                        for (uint t = 0; t < pc.shared_dim/TILE_SIZE; t++) {
                            // each thread loads one element of shared A and shared B for 2 total memory accesses for each tile we load to calculate a tile of C 
                            // so 2* pc.shared_dim/TILE_SIZE memory loads per thread
                            // =====
                            // 
                            tile_A[thread_row][thread_col] = a.data[c_row*pc.shared_dim + t*TILE_SIZE+thread_col];
                            tile_B[thread_row][thread_col] = b.data[(t*TILE_SIZE+thread_row)*pc.ret_cols + c_col];
                            // we need to make sure the shared memory is up to date but since updating is split amongst threads in the group we need this barrier to synchronize them
                            // preventing them from moving ahead before their peers have finished
                            barrier();
                            // each thread accesses the data it needs from the data it and it's peers have collaboratively loaded above
                            // each performs the 'mini' dot product in this loop which makes up this tile's porting of the overall 'full' dot product of the entire row of A (#c_row) and corresponding entire column of B (#c_col)
                            for (uint i = 0; i < TILE_SIZE; i++){
                                acc += tile_A[thread_row][i] * tile_B[i][thread_col];
                            }
                            // before we start to make changes to the shared memory again we need to make sure all threads in the group are done using it in this last loop
                            barrier();
                        }
    
                        c.data[c_row*pc.ret_cols+c_col] = acc;
                    }
                ",

            }
        }
        let shader = cs::Shader::load(device.clone()).unwrap();
        // The `vulkano_shaders::shaders!` macro generates a struct with the correct representation of the push constants struct specified in the shader.
        // Here we create an instance of the generated struct.
        let push_constants = cs::ty::PushConstantData {
            shared_dim: 64,
            ret_cols: 64,
            ret_rows: 64,
        };
        (
            Arc::new(
                ComputePipeline::new(device.clone(), &shader.main_entry_point(), &(), None)
                    .unwrap(),
            ),
            push_constants,
        )
    };

    // We start by creating the buffer that will store the data.
    let data_buffer0 = {
        // Iterator that produces the data.
        let data_iter = (0..4096u32).map(|n| n);
        // Builds the buffer and fills it with this iterator.
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, data_iter)
            .unwrap()
    };
    // We start by creating the buffer that will store the data.
    let data_buffer1 = {
        // Iterator that produces the data.
        let data_iter = (0..4096u32).map(|n| n);
        // Builds the buffer and fills it with this iterator.
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, data_iter)
            .unwrap()
    };
    let ret_buffer = {
        // Iterator that produces the data.
        let data_iter = (0..4096u32).map(|_| 0 as u32);
        // Builds the buffer and fills it with this iterator.
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, data_iter)
            .unwrap()
    };

    // In order to let the shader access the buffer, we need to build a *descriptor set* that
    // contains the buffer.
    //
    // The resources that we bind to the descriptor set must match the resources expected by the
    // pipeline which we pass as the first parameter.
    //
    // If you want to run the pipeline on multiple different buffers, you need to create multiple
    // descriptor sets that each contain the buffer you want to run the shader on.
    let layout = pipeline.layout().descriptor_set_layout(0).unwrap();
    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(data_buffer0.clone())
            .unwrap()
            .add_buffer(data_buffer1.clone())
            .unwrap()
            .add_buffer(ret_buffer.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    // In order to execute our operation, we have to build a command buffer.
    let mut builder =
        AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap();
    builder
        // The command buffer only does one thing: execute the compute pipeline.
        // This is called a *dispatch* operation.
        //
        // Note that we clone the pipeline and the set. Since they are both wrapped around an
        // `Arc`, this only clones the `Arc` and not the whole pipeline or set (which aren't
        // cloneable anyway). In this example we would avoid cloning them since this is the last
        // time we use them, but in a real code you would probably need to clone them.
        .dispatch([4, 4, 1], pipeline.clone(), set.clone(), push_constants)
        .unwrap();
    // Finish building the command buffer by calling `build`.
    let command_buffer = builder.build().unwrap();

    // Let's execute this command buffer now.
    // To do so, we TODO: this is a bit clumsy, probably needs a shortcut
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        // This line instructs the GPU to signal a *fence* once the command buffer has finished
        // execution. A fence is a Vulkan object that allows the CPU to know when the GPU has
        // reached a certain point.
        // We need to signal a fence here because below we want to block the CPU until the GPU has
        // reached that point in the execution.
        .then_signal_fence_and_flush()
        .unwrap();

    // Blocks execution until the GPU has finished the operation. This method only exists on the
    // future that corresponds to a signalled fence. In other words, this method wouldn't be
    // available if we didn't call `.then_signal_fence_and_flush()` earlier.
    // The `None` parameter is an optional timeout.
    //
    // Note however that dropping the `future` variable (with `drop(future)` for example) would
    // block execution as well, and this would be the case even if we didn't call
    // `.then_signal_fence_and_flush()`.
    // Therefore the actual point of calling `.then_signal_fence_and_flush()` and `.wait()` is to
    // make things more explicit. In the future, if the Rust language gets linear types vulkano may
    // get modified so that only fence-signalled futures can get destroyed like this.
    future.wait(None).unwrap();

    // Now that the GPU is done, the content of the buffer should have been modified. Let's
    // check it out.
    // The call to `read()` would return an error if the buffer was still in use by the GPU.
    let data_buffer_content = ret_buffer.read().unwrap();
    println!("{:?}", &data_buffer_content[556..597]);
    // for n in 0..65536u32 {
    //     assert_eq!(data_buffer_content[n as usize], n * 12);
    // }
}
