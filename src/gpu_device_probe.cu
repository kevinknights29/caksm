/**
 * @file gpu_device_probe.cu
 * @brief Report static CUDA capabilities that bound copy and kernel overlap.
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        const cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error %s at %s:%d -- %s\n",                    \
                         cudaGetErrorName(err_), __FILE__, __LINE__,                  \
                         cudaGetErrorString(err_));                                   \
            std::exit(EXIT_FAILURE);                                                  \
        }                                                                             \
    } while (0)

struct Args {
    int device = -1;
    std::string csv_path;
};

[[nodiscard]] Args parse_args(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (++i >= argc) {
                std::fprintf(stderr, "Missing value after %s\n", arg.c_str());
                std::exit(EXIT_FAILURE);
            }
            return argv[i];
        };
        if (arg == "--device") {
            const std::string value = next();
            args.device = value == "all" ? -1 : std::stoi(value);
        } else if (arg == "--csv") {
            args.csv_path = next();
        } else if (arg == "--help") {
            std::printf("Usage: gpu-device-probe [--device all|N] [--csv PATH]\n");
            std::exit(EXIT_SUCCESS);
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            std::exit(EXIT_FAILURE);
        }
    }
    return args;
}

void write_csv_name(std::FILE* file, const char* name)
{
    std::fputc('"', file);
    for (const char* p = name; *p != '\0'; ++p) {
        if (*p == '"') std::fputc('"', file);
        std::fputc(*p, file);
    }
    std::fputc('"', file);
}

} // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        std::fprintf(stderr, "No CUDA devices are visible.\n");
        return EXIT_FAILURE;
    }
    if (args.device >= count) {
        std::fprintf(stderr, "Device %d is outside the visible range [0, %d).\n",
                     args.device, count);
        return EXIT_FAILURE;
    }

    std::FILE* csv = nullptr;
    if (!args.csv_path.empty()) {
        csv = std::fopen(args.csv_path.c_str(), "w");
        if (csv == nullptr) {
            std::fprintf(stderr, "Cannot open CSV: %s\n", args.csv_path.c_str());
            return EXIT_FAILURE;
        }
        std::fprintf(csv, "device,device_name,async_engine_count,concurrent_kernels,"
                          "device_overlap,unified_addressing\n");
    }

    const int begin = args.device < 0 ? 0 : args.device;
    const int end = args.device < 0 ? count : args.device + 1;
    for (int device = begin; device < end; ++device) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::printf(
            "GPU_CAPABILITIES device=%d name=\"%s\" async_engine_count=%d "
            "concurrent_kernels=%d device_overlap=%d unified_addressing=%d\n",
            device, prop.name, prop.asyncEngineCount, prop.concurrentKernels,
            prop.deviceOverlap, prop.unifiedAddressing);
        if (csv != nullptr) {
            std::fprintf(csv, "%d,", device);
            write_csv_name(csv, prop.name);
            std::fprintf(csv, ",%d,%d,%d,%d\n", prop.asyncEngineCount,
                         prop.concurrentKernels, prop.deviceOverlap,
                         prop.unifiedAddressing);
        }
    }

    if (csv != nullptr) {
        std::fclose(csv);
        std::printf("GPU_CAPABILITIES_CSV path=%s devices=%d\n",
                    args.csv_path.c_str(), end - begin);
    }
    return EXIT_SUCCESS;
}
