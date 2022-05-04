use std::sync::Arc;
use std::alloc::Layout;

use log::*;

use winit::window::WindowBuilder;
use winit::event_loop::EventLoop;

use vulkano::instance::{
	Instance,
	PhysicalDevice,
};
use vulkano::swapchain::{
	Swapchain,
	Surface,
	ColorSpace,
	FullscreenExclusive,
	PresentMode,
	SurfaceTransform,
};
use vulkano::device::{
	Device,
	DeviceExtensions,
	Queue,
};
use vulkano::image::{
	SwapchainImage,
	ImageUsage,
};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::framebuffer::Subpass;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::pipeline::vertex::SingleBufferDefinition;

use vulkano_win::VkSurfaceBuild;

use crate::glsl_shaders::*;

pub fn init_vlk(event_loop: &EventLoop<()>) -> (
	Arc<Surface<winit::window::Window>>,
	Arc<Swapchain<winit::window::Window>>,
	Vec<Arc<SwapchainImage<winit::window::Window>>>,
	Arc<Queue>,
	Arc<Device>
	) {
	let required_extensions = vulkano_win::required_extensions();
	let instance = Instance::new(None, &required_extensions, None).unwrap();
	let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

	info!(
		"Using device: {} (type: {:?})",
		physical.name(),
		physical.ty()
	);

	let surface = WindowBuilder::new()
		.build_vk_surface(event_loop, instance.clone())
		.unwrap();

	let queue_family = physical
		.queue_families()
		.find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
		.unwrap();

	let device_ext = DeviceExtensions {
		khr_swapchain: true,
		..DeviceExtensions::none()
	};
	let (device, mut queues) = Device::new(
		physical,
		physical.supported_features(),
		&device_ext,
		[(queue_family, 0.5)].iter().cloned(),
	)
	.unwrap();

	let queue = queues.next().unwrap();

	let caps = surface.capabilities(physical).unwrap();
	let alpha = caps.supported_composite_alpha.iter().next().unwrap();
	let format = caps.supported_formats[0].0;
	let dimensions: [u32; 2] = surface.window().inner_size().into();

	let (swapchain, images) = {
		Swapchain::new(
			device.clone(),
			surface.clone(),
			caps.min_image_count,
			format,
			dimensions,
			1,
			ImageUsage::color_attachment(),
			&queue,
			SurfaceTransform::Identity,
			alpha,
			PresentMode::Fifo,
			FullscreenExclusive::Default,
			true,
			ColorSpace::SrgbNonLinear,
		)
		.unwrap()
	};

	return (surface, swapchain, images, queue, device);
}
