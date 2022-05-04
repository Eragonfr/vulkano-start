use log::*;

use vulkano::buffer::CpuBufferPool;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, SubpassContents};
use vulkano::framebuffer::Subpass;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::swapchain;
use vulkano::swapchain::{
	AcquireError,
	SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

mod glsl_shaders;
mod win_utils;
mod vulk_utils;

use crate::glsl_shaders::*;
use crate::win_utils::window_size_dependent_setup;
use crate::vulk_utils::init_vlk;

#[derive(Default, Debug, Clone)]
struct Vertex {
	position: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position);

fn main() {
	dotenv::dotenv().expect("Failed to load .env file");
	env_logger::builder().format_timestamp(None).init();

	let event_loop = EventLoop::new();

	let (surface, mut swapchain, images, queue, device) = init_vlk(&event_loop);

	info!("Vulkan init ended succesifully");

	// Vertex Buffer Pool
	let buffer_pool: CpuBufferPool<Vertex> = CpuBufferPool::vertex_buffer(device.clone());

	let vs = vs::Shader::load(device.clone()).unwrap();
	let fs = fs::Shader::load(device.clone()).unwrap();

	let render_pass = Arc::new(
		vulkano::single_pass_renderpass!(
			device.clone(),
			attachments: {
				color: {
					load: Clear,
					store: Store,
					format: swapchain.format(),
					samples: 1,
				}
			},
			pass: {
				color: [color],
				depth_stencil: {}
			}
		)
		.unwrap(),
	);

	let pipeline = Arc::new(
		GraphicsPipeline::start()
			.vertex_input_single_buffer()
			.vertex_shader(vs.main_entry_point(), ())
			.triangle_list()
			.viewports_dynamic_scissors_irrelevant(1)
			.fragment_shader(fs.main_entry_point(), ())
			.render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
			.build(device.clone())
			.unwrap(),
	);

	let mut dynamic_state = DynamicState {
		line_width: None,
		viewports: None,
		scissors: None,
		compare_mask: None,
		write_mask: None,
		reference: None,
	};
	let mut framebuffers =
		window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);
	let mut recreate_swapchain = false;
	let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

	event_loop.run(move |event, _, control_flow| {
		match event {
			Event::WindowEvent {
				event: WindowEvent::CloseRequested,
				..
			} => {
				*control_flow = ControlFlow::Exit;
			}
			Event::WindowEvent {
				event: WindowEvent::Resized(_),
				..
			} => {
				recreate_swapchain = true;
			}
			Event::RedrawEventsCleared => {
				previous_frame_end.as_mut().unwrap().cleanup_finished();

				if recreate_swapchain {
					let dimensions: [u32; 2] = surface.window().inner_size().into();
					let (new_swapchain, new_images) =
						match swapchain.recreate_with_dimensions(dimensions) {
							Ok(r) => r,
							Err(SwapchainCreationError::UnsupportedDimensions) => return,
							Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
						};

					swapchain = new_swapchain;
					framebuffers = window_size_dependent_setup(
						&new_images,
						render_pass.clone(),
						&mut dynamic_state,
					);
					recreate_swapchain = false;
				}

				let (image_num, suboptimal, acquire_future) =
					match swapchain::acquire_next_image(swapchain.clone(), None) {
						Ok(r) => r,
						Err(AcquireError::OutOfDate) => {
							recreate_swapchain = true;
							return;
						}
						Err(e) => panic!("Failed to acquire next image: {:?}", e),
					};

				if suboptimal {
					recreate_swapchain = true;
				}

				let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];

				// Rotate once (PI*2) every 5 seconds
				let elapsed = SystemTime::now()
					.duration_since(UNIX_EPOCH)
					.unwrap()
					.as_secs_f64();
				const DURATION: f64 = 5.0;
				let remainder = elapsed.rem_euclid(DURATION);
				let delta = (remainder / DURATION) as f32;
				let angle = delta * std::f32::consts::PI * 2.0;
				const RADIUS: f32 = 0.5;
				// 120Degree offset in radians
				const ANGLE_OFFSET: f32 = (std::f32::consts::PI * 2.0) / 3.0;
				// Calculate vertices
				let data = [
					Vertex {
						position: [angle.cos() * RADIUS, angle.sin() * RADIUS],
					},
					Vertex {
						position: [
							(angle + ANGLE_OFFSET).cos() * RADIUS,
							(angle + ANGLE_OFFSET).sin() * RADIUS,
						],
					},
					Vertex {
						position: [
							(angle - ANGLE_OFFSET).cos() * RADIUS,
							(angle - ANGLE_OFFSET).sin() * RADIUS,
						],
					},
				];

				// Allocate a new chunk from buffer_pool
				let buffer = buffer_pool.chunk(data.to_vec()).unwrap();
				let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
					device.clone(),
					queue.family(),
				)
				.unwrap();
				builder
					.begin_render_pass(
						framebuffers[image_num].clone(),
						SubpassContents::Inline,
						clear_values,
					)
					.unwrap()
					// Draw our buffer
					.draw(pipeline.clone(), &dynamic_state, buffer, (), (), vec![])
					.unwrap()
					.end_render_pass()
					.unwrap();
				let command_buffer = builder.build().unwrap();

				let future = previous_frame_end
					.take()
					.unwrap()
					.join(acquire_future)
					.then_execute(queue.clone(), command_buffer)
					.unwrap()
					.then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
					.then_signal_fence_and_flush();

				match future {
					Ok(future) => {
						previous_frame_end = Some(Box::new(future) as Box<_>);
					}
					Err(FlushError::OutOfDate) => {
						recreate_swapchain = true;
						previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
					}
					Err(e) => {
						println!("Failed to flush future: {:?}", e);
						previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
					}
				}
			}
			_ => (),
		}
	});
}
