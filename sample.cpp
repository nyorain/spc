#include "spirv_cross_parsed_ir.hpp"
#include "spirv_parser.hpp"
#include <cstdio>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <span>

namespace spc = spirv_cross;

template<typename F, bool OnSuccess = true, bool OnException = true>
class ScopeGuard {
public:
	static_assert(OnSuccess || OnException);

public:
	ScopeGuard(F&& func) :
		func_(std::forward<F>(func)),
		exceptions_(std::uncaught_exceptions()) {}

	ScopeGuard(ScopeGuard&&) = delete;
	ScopeGuard& operator =(ScopeGuard&&) = delete;

	~ScopeGuard() noexcept {
		if(exceptions_ == -1) {
			return;
		}

		try {
			auto thrown = exceptions_ < std::uncaught_exceptions();
			if((OnSuccess && !thrown) || (OnException && thrown)) {
				func_();
			}
		} catch(const std::exception& err) {
			printf("~ScopeGuard: caught exception while unwinding: %s\n", err.what());
		} catch(...) {
			printf("~ScopeGuard: caught non-exception while unwinding\n");
		}
	}

	void unset() { exceptions_ = -1; }

protected:
	F func_;
	int exceptions_;
};

// Returns ceil(num / denom), efficiently, only using integer division.
inline constexpr unsigned ceilDivide(unsigned num, unsigned denom) {
	return (num + denom - 1) / denom;
}

template<typename C>
C readFile(const char* path, bool binary) {
	assert(path);
	errno = 0;

	auto *f = std::fopen(path, binary ? "rb" : "r");
	if(!f) {
		printf("Could not open '%s' for reading: %s\n", path, std::strerror(errno));
		return {};
	}


	auto ret = std::fseek(f, 0, SEEK_END);
	if(ret != 0) {
		printf("fseek on '%s' failed: %s\n", path, std::strerror(errno));
		return {};
	}

	auto fsize = std::ftell(f);
	if(fsize < 0) {
		printf("ftell on '%s' failed: %s\n", path, std::strerror(errno));
		return {};
	}

	ret = std::fseek(f, 0, SEEK_SET);
	if(ret != 0) {
		printf("second fseek on '%s' failed: %s\n", path, std::strerror(errno));
		return {};
	}

	assert(fsize % sizeof(typename C::value_type) == 0);

	C buffer(ceilDivide(fsize, sizeof(typename C::value_type)), {});
	ret = std::fread(buffer.data(), 1, fsize, f);
	if(ret != fsize) {
		printf("fread on '%s' failed: %s\n", path, std::strerror(errno));
		return {};
	}

	return buffer;
}

using u32 = std::uint32_t;
using u16 = std::uint32_t;
template std::vector<u32> readFile<std::vector<u32>>(const char*, bool);

void writeFile(const char* path, std::span<const std::byte> buffer, bool binary) {
	assert(path);
	errno = 0;

	auto* f = std::fopen(path, binary ? "wb" : "w");
	if(!f) {
		// dlg_error("Could not open '{}' for writing: {}", path, std::strerror(errno));
		printf("could not open file for writing\n");
		return;
	}

	auto ret = std::fwrite(buffer.data(), 1, buffer.size(), f);
	if(ret != buffer.size()) {
		// dlg_error("fwrite on '{}' failed: {}", path, std::strerror(errno));
		printf("fwrite failed\n");
	}

	std::fclose(f);
}

struct InstrBuilder {
	spv::Op op;
	std::vector<u32> vals {0}; // first val is reserved

	void insert(std::vector<u32>& dst, u32 off) {
		assert(dst.size() >= off);
		vals[0] = u16(vals.size()) << 16 | u16(op);
		dst.insert(dst.begin() + off, vals.begin(), vals.end());
		vals.clear();
	}

	template<typename T>
	std::enable_if_t<std::is_integral_v<T> || std::is_enum_v<T>, InstrBuilder&>
	push(T val) {
		static_assert(sizeof(T) <= sizeof(u32));
		vals.push_back(u32(val));
		return *this;
	}

	InstrBuilder& push(std::string_view val) {
		for(auto i = 0u; i < val.size(); i += 4) {
			u32 ret = val[i];
			if(i + 1 < val.size()) ret |= val[i + 1] << 8;
			if(i + 2 < val.size()) ret |= val[i + 2] << 16;
			if(i + 3 < val.size()) ret |= val[i + 3] << 24;
			vals.push_back(ret);
		}

		return *this;
	}

	~InstrBuilder() {
		assert(vals.empty());
	}
};

void outputPatched(const spc::ParsedIR& ir, u32 file, u32 line) {
	auto copy = ir.spirv;

	// set new memory addressing model
	auto& addressing = copy[ir.section_offsets.named.memModel + 1];
	if(addressing != u32(spv::AddressingModelPhysicalStorageBuffer64)) {
		assert(addressing == u32(spv::AddressingModelLogical));
		addressing = u32(spv::AddressingModelPhysicalStorageBuffer64);
	}

	// add extension
	InstrBuilder{spv::OpExtension}
		.push("SPV_KHR_physical_storage_buffer")
		.insert(copy, ir.section_offsets.named.exts);

	// add capability
	InstrBuilder{spv::OpCapability}
		.push(spv::CapabilityPhysicalStorageBufferAddresses)
		.insert(copy, ir.section_offsets.named.caps);

	// find target position
	assert(file < ir.sources.size());
	auto& source = ir.sources[file];
	auto cmp = [](auto& a, auto& b) {
		return a.line < b;
	};
	auto lb = std::lower_bound(source.line_markers.begin(),
		source.line_markers.end(), line, cmp);
	if(lb == source.line_markers.end()) {
		printf("no matching line found\n");
		return;
	}

	if(lb->line != line) {
		printf("no exact match found: %d vs %d\n", line, lb->line);
	}

	assert(lb->function);
	const auto& name = ir.get_name(lb->function->self);
	printf("in function %s\n", name.c_str());

	for (auto& varID : lb->function->local_variables) {
		// auto& var = ir.ids[varID].get<spc::SPIRVariable>();
		printf(" >> var %s\n", ir.get_name(varID).c_str());
	}

	// rough idea:
	// - build buffmt types for all the variables.
	//   OpVariable is always a pointer, basically use the pointed-to type
	//   Also, pre-filter, discard variables we cannot use
	// - build struct from it with buffmt
	//   (ugh, alignment. just require scalar for now?)
	// - declare that struct type in spirv [patch]
	// - declare OpTypePointer TP to that struct with PhysicalStorageBuffer [patch]
	// - declare A, uint2 OpConstantComposite with address [patch]
	// - at dst: OpBitcast B from A to type TP
	//   construct struct C via OpCompositeConstruct
	//   OpStore C to B
	// - somehow need to increase id bound.
	// - conditions: need to create new blocks, split up
	//   OpIEqual (if we want to do compare that)
	//   OpAll
	//   OpSelectionMerge
	//   OpBranchConditional
	//   OpLabel [new 1]
	//   ... (see above)
	//   OpBranch
	//   OpLabel [new 2]
	// - possibly need to declare builtins that shader did previously not need.
	//   configuring that in UI will require some thought

	writeFile("out.spv", std::as_bytes(std::span(copy)), true);
}

int main(int argc, const char** argv) {
	assert(argc > 1);
	std::vector<u32> spirv = readFile<std::vector<u32>>(argv[1], true);

	spc::Parser parser(std::move(spirv));
	parser.parse();
	auto& ir = parser.get_parsed_ir();

	outputPatched(ir, 0, 20);
}
