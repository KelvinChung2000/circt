// A class containing a bunch of syntactic sugar to reduce builder function
// verbosity.
// @todo: should be moved to support.

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include <memory>

using namespace mlir;
using namespace circt;
using namespace circt::hw;

using NameUniquer = std::function<std::string(Operation *)>;

struct RTLBuilder {
  RTLBuilder(hw::ModulePortInfo info, OpBuilder &builder, Location loc,
             Value clk = Value(), Value rst = Value())
      : info(std::move(info)), b(builder), loc(loc), clk(clk), rst(rst) {}

  Value constant(const APInt &apv, std::optional<StringRef> name = {});
  Value constant(unsigned width, int64_t value,
                 std::optional<StringRef> name = {});
  std::pair<Value, Value> wrap(Value data, Value valid,
                               std::optional<StringRef> name = {});
  std::pair<Value, Value> unwrap(Value channel, Value ready,
                                 std::optional<StringRef> name = {});
  Value reg(StringRef name, Value in, Value rstValue, Value clk = Value(),
            Value rst = Value());
  Value cmp(Value lhs, Value rhs, comb::ICmpPredicate predicate,
            std::optional<StringRef> name = {});
  Value buildNamedOp(llvm::function_ref<Value()> f,
                     std::optional<StringRef> name);
  Value bAnd(ValueRange values, std::optional<StringRef> name = {});
  Value bOr(ValueRange values, std::optional<StringRef> name = {});
  Value bNot(Value value, std::optional<StringRef> name = {});
  Value shl(Value value, Value shift, std::optional<StringRef> name = {});
  Value concat(ValueRange values, std::optional<StringRef> name = {});
  Value pack(ValueRange values, Type structType = Type(),
             std::optional<StringRef> name = {});
  ValueRange unpack(Value value);
  llvm::SmallVector<Value> toBits(Value v, std::optional<StringRef> name = {});
  Value rOr(Value v, std::optional<StringRef> name = {});
  Value extract(Value v, unsigned lo, unsigned hi,
                std::optional<StringRef> name = {});
  Value truncate(Value value, unsigned width,
                 std::optional<StringRef> name = {});
  Value zext(Value value, unsigned outWidth,
             std::optional<StringRef> name = {});
  Value sext(Value value, unsigned outWidth,
             std::optional<StringRef> name = {});
  Value bit(Value v, unsigned index, std::optional<StringRef> name = {});
  Value arrayCreate(ValueRange values, std::optional<StringRef> name = {});
  Value arrayGet(Value array, Value index, std::optional<StringRef> name = {});
  Value mux(Value index, ValueRange values, std::optional<StringRef> name = {});
  Value ohMux(Value index, ValueRange inputs);
  hw::ModulePortInfo info;
  OpBuilder &b;
  Location loc;
  Value clk, rst;
  DenseMap<APInt, Value> constants;
};