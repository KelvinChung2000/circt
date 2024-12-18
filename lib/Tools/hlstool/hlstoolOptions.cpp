#include "circt/Tools/hlstool/hlstool.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace circt::hlstool;

static cl::OptionCategory GeneralHLSCategory("HLS Options 2");

namespace {
struct hlstoolOptions {
  // cl::opt<std::string> kernel{"kernel", cl::desc("Name of the kernel
  // function"),
  //                             cl::init(""), cl::cat(GeneralHLSCategory)};
};

} // namespace

ManagedStatic<hlstoolOptions> hlstoolcl;
void hlstool::registerHLSToolOptions() { *hlstoolcl; }

hlsFlags::hlsFlags() {
  // kernel = hlstoolcl->kernel;
}