#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

namespace kda_scale {
using namespace nvinfer1;
ILoggerFinder* loggerFinder = nullptr;
constexpr char kName[] = "KdaScale";
constexpr char kVersion[] = "1";
constexpr char kNamespace[] = "";

int32_t fail(char const* message) noexcept {
    if (loggerFinder && loggerFinder->findLogger())
        loggerFinder->findLogger()->log(ILogger::Severity::kERROR, message);
    std::fprintf(stderr, "KdaScale: %s\n", message);
    return 1;
}

__global__ void scaleKernel(float const* input, float* output, int64_t count, float scale) {
    for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += int64_t(blockDim.x) * gridDim.x) {
        output[index] = input[index] * scale;
    }
}

class ScalePlugin final : public IPluginV3, public IPluginV3OneCore,
                          public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    explicit ScalePlugin(float scale) noexcept : mScale(scale),
        mField{"scale", &mScale, PluginFieldType::kFLOAT32, 1}, mFields{1, &mField} {}

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        fail("Unknown capability");
        return nullptr;
    }
    IPluginV3* clone() noexcept override {
        auto* result = new (std::nothrow) ScalePlugin(mScale);
        if (!result) fail("Plugin allocation failed");
        return result;
    }
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return kNamespace; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(DataType* outputs, int32_t nbOutputs,
                              DataType const* inputs, int32_t nbInputs) const noexcept override {
        if (nbInputs != 1 || nbOutputs != 1 || inputs[0] != DataType::kFLOAT)
            return fail("Expected one FP32 input and output");
        outputs[0] = inputs[0];
        return 0;
    }
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
                           DimsExprs const*, int32_t nbShapeInputs, DimsExprs* outputs,
                           int32_t nbOutputs, IExprBuilder&) noexcept override {
        if (nbInputs != 1 || nbOutputs != 1 || nbShapeInputs != 0)
            return fail("Invalid shape interface");
        outputs[0] = inputs[0];
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* tensors,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override {
        return nbInputs == 1 && nbOutputs == 1 && pos >= 0 && pos < 2
            && tensors[pos].desc.type == DataType::kFLOAT
            && tensors[pos].desc.format == TensorFormat::kLINEAR;
    }
    int32_t configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                            DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override {
        if (nbInputs != 1 || nbOutputs != 1 || inputs[0].desc.type != DataType::kFLOAT
            || outputs[0].desc.type != DataType::kFLOAT
            || inputs[0].desc.format != TensorFormat::kLINEAR
            || outputs[0].desc.format != TensorFormat::kLINEAR)
            return fail("Unsupported configuration");
        return 0;
    }
    int32_t onShapeChange(PluginTensorDesc const* inputs, int32_t nbInputs,
                         PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override {
        if (nbInputs != 1 || nbOutputs != 1 || inputs[0].type != DataType::kFLOAT
            || outputs[0].type != DataType::kFLOAT
            || inputs[0].format != TensorFormat::kLINEAR || outputs[0].format != TensorFormat::kLINEAR
            || inputs[0].dims.nbDims < 0 || inputs[0].dims.nbDims != outputs[0].dims.nbDims)
            return fail("Invalid runtime tensors");
        for (int32_t axis = 0; axis < inputs[0].dims.nbDims; ++axis)
            if (inputs[0].dims.d[axis] < 0 || inputs[0].dims.d[axis] != outputs[0].dims.d[axis])
                return fail("Invalid runtime dimensions");
        return 0;
    }
    int32_t enqueue(PluginTensorDesc const* inputs, PluginTensorDesc const*,
                    void const* const* input, void* const* output, void*, cudaStream_t stream) noexcept override {
        int64_t count = 1;
        for (int32_t axis = 0; axis < inputs[0].dims.nbDims; ++axis) {
            int64_t dim = inputs[0].dims.d[axis];
            if (dim < 0 || (dim && count > std::numeric_limits<int64_t>::max() / dim))
                return fail("Element count overflow or unresolved dimension");
            count *= dim;
        }
        if (count == 0) return 0;
        int64_t blocks = count / 256 + (count % 256 != 0);
        if (blocks > 65535) blocks = 65535;
        scaleKernel<<<static_cast<int>(blocks), 256, 0, stream>>>(
            static_cast<float const*>(input[0]), static_cast<float*>(output[0]), count, mScale);
        cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess) return fail(cudaGetErrorString(result));
        return 0;
    }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFields; }
private:
    float mScale;
    PluginField mField;
    PluginFieldCollection mFields;
};

class ScaleCreator final : public IPluginCreatorV3One {
public:
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return kNamespace; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFields; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fields,
                           TensorRTPhase) noexcept override {
        if (!fields || fields->nbFields != 1 || !fields->fields) {
            fail("Exactly one scale field is required");
            return nullptr;
        }
        auto const& field = fields->fields[0];
        if (!field.name || std::strcmp(field.name, "scale") || !field.data
            || field.type != PluginFieldType::kFLOAT32 || field.length != 1) {
            fail("Invalid scale field");
            return nullptr;
        }
        float scale;
        std::memcpy(&scale, field.data, sizeof(scale));
        if (!std::isfinite(scale)) {
            fail("Scale must be finite");
            return nullptr;
        }
        auto* result = new (std::nothrow) ScalePlugin(scale);
        if (!result) fail("Plugin allocation failed");
        return result;
    }
private:
    PluginField mField{"scale", nullptr, PluginFieldType::kFLOAT32, 1};
    PluginFieldCollection mFields{1, &mField};
};
}

extern "C" nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& count) {
    static kda_scale::ScaleCreator creator;
    static nvinfer1::IPluginCreatorInterface* creators[] = {&creator};
    count = 1;
    return creators;
}

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder* finder) {
    kda_scale::loggerFinder = finder;
}
