project('spc', ['c', 'cpp'],
	version: '1.3.290',
	default_options: [
		'warning_level=3',
		'cpp_std=c++20',
		'default_library=static',
	]
)

inc = include_directories('.')

src = files(
	'spirv_cfg.cpp',
	'spirv_cross.cpp',
	'spirv_parser.cpp',
	'spirv_cross_parsed_ir.cpp',

	# mostly for comfort
	'spirv.h',
	'spirv.hpp',
	'spirv_cfg.hpp',
	'spirv_common.hpp',
	'spirv_cross.hpp',
	'spirv_cross_containers.hpp',
	'spirv_cross_error_handling.hpp',
	'spirv_cross_parsed_ir.hpp',
	'spirv_parser.hpp',
)

deps = []

libspc = library('spc', src,
	include_directories: inc,
	dependencies: deps)

spc_dep = declare_dependency(
	include_directories: inc,
	compile_args: [],
	link_with: [libspc],
	dependencies: deps)

sample = executable('sample', 'sample.cpp',
	dependencies: [spc_dep])
