#include "circt/Support/RTLBuilder.h"
#include "circt/Conversion/HandshakeToHW.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace circt::hw;

using NameUniquer = std::function<std::string(Operation *)>;

static Type tupleToStruct(TypeRange types) {
  return toValidType(mlir::TupleType::get(types[0].getContext(), types));
}

Value RTLBuilder::constant(const APInt &apv, std::optional<StringRef> name) {
  // Cannot use zero-width APInt's in DenseMap's, see
  // https://github.com/llvm/llvm-project/issues/58013
  bool isZeroWidth = apv.getBitWidth() == 0;
  if (!isZeroWidth) {
    auto it = constants.find(apv);
    if (it != constants.end())
      return it->second;
  }

  auto cval = b.create<hw::ConstantOp>(loc, apv);
  if (!isZeroWidth)
    constants[apv] = cval;
  return cval;
}

Value RTLBuilder::constant(unsigned width, int64_t value,
                           std::optional<StringRef> name) {
  return constant(APInt(width, value));
}

std::pair<Value, Value> RTLBuilder::wrap(Value data, Value valid,
                                         std::optional<StringRef> name) {
  auto wrapOp = b.create<esi::WrapValidReadyOp>(loc, data, valid);
  return {wrapOp.getResult(0), wrapOp.getResult(1)};
}

std::pair<Value, Value> RTLBuilder::unwrap(Value channel, Value ready,
                                           std::optional<StringRef> name) {
  auto unwrapOp = b.create<esi::UnwrapValidReadyOp>(loc, channel, ready);
  return {unwrapOp.getResult(0), unwrapOp.getResult(1)};
}

// Various syntactic sugar functions.
Value RTLBuilder::reg(StringRef name, Value in, Value rstValue, Value clk,
                      Value rst) {
  Value resolvedClk = clk ? clk : this->clk;
  Value resolvedRst = rst ? rst : this->rst;
  assert(resolvedClk && "No global clock provided to this RTLBuilder - a clock "
                        "signal must be provided to the reg(...) function.");
  assert(resolvedRst && "No global reset provided to this RTLBuilder - a reset "
                        "signal must be provided to the reg(...) function.");

  return b.create<seq::CompRegOp>(loc, in, resolvedClk, resolvedRst, rstValue,
                                  name);
}

Value RTLBuilder::cmp(Value lhs, Value rhs, comb::ICmpPredicate predicate,
                      std::optional<StringRef> name) {
  return b.create<comb::ICmpOp>(loc, predicate, lhs, rhs);
}

Value RTLBuilder::buildNamedOp(llvm::function_ref<Value()> f,
                               std::optional<StringRef> name) {
  Value v = f();
  StringAttr nameAttr;
  Operation *op = v.getDefiningOp();
  if (name.has_value()) {
    op->setAttr("sv.namehint", b.getStringAttr(*name));
    nameAttr = b.getStringAttr(*name);
  }
  return v;
}

// Bitwise 'and'.
Value RTLBuilder::bAnd(ValueRange values, std::optional<StringRef> name) {
  return buildNamedOp(
      [&]() { return b.create<comb::AndOp>(loc, values, false); }, name);
}

Value RTLBuilder::bOr(ValueRange values, std::optional<StringRef> name) {
  return buildNamedOp(
      [&]() { return b.create<comb::OrOp>(loc, values, false); }, name);
}

// Bitwise 'not'.
Value RTLBuilder::bNot(Value value, std::optional<StringRef> name) {
  auto allOnes = constant(value.getType().getIntOrFloatBitWidth(), -1);
  std::string inferedName;
  if (!name) {
    // Try to create a name from the input value.
    if (auto valueName =
            value.getDefiningOp()->getAttrOfType<StringAttr>("sv.namehint")) {
      inferedName = ("not_" + valueName.getValue()).str();
      name = inferedName;
    }
  }

  return buildNamedOp(
      [&]() { return b.create<comb::XorOp>(loc, value, allOnes); }, name);

  return b.createOrFold<comb::XorOp>(loc, value, allOnes, false);
}

Value RTLBuilder::shl(Value value, Value shift, std::optional<StringRef> name) {
  return buildNamedOp(
      [&]() { return b.create<comb::ShlOp>(loc, value, shift); }, name);
}

Value RTLBuilder::concat(ValueRange values, std::optional<StringRef> name) {
  return buildNamedOp([&]() { return b.create<comb::ConcatOp>(loc, values); },
                      name);
}

// Packs a list of values into a hw.struct.
Value RTLBuilder::pack(ValueRange values, Type structType,
                       std::optional<StringRef> name) {
  if (!structType)
    structType = tupleToStruct(values.getTypes());
  return buildNamedOp(
      [&]() { return b.create<hw::StructCreateOp>(loc, structType, values); },
      name);
}

// Unpacks a hw.struct into a list of values.
ValueRange RTLBuilder::unpack(Value value) {
  auto structType = cast<hw::StructType>(value.getType());
  llvm::SmallVector<Type> innerTypes;
  structType.getInnerTypes(innerTypes);
  return b.create<hw::StructExplodeOp>(loc, innerTypes, value).getResults();
}

llvm::SmallVector<Value> RTLBuilder::toBits(Value v,
                                            std::optional<StringRef> name) {
  llvm::SmallVector<Value> bits;
  for (unsigned i = 0, e = v.getType().getIntOrFloatBitWidth(); i != e; ++i)
    bits.push_back(b.create<comb::ExtractOp>(loc, v, i, /*bitWidth=*/1));
  return bits;
}

// OR-reduction of the bits in 'v'.
Value RTLBuilder::rOr(Value v, std::optional<StringRef> name) {
  return buildNamedOp([&]() { return bOr(toBits(v)); }, name);
}

// Extract bits v[hi:lo] (inclusive).
Value RTLBuilder::extract(Value v, unsigned lo, unsigned hi,
                          std::optional<StringRef> name) {
  unsigned width = hi - lo + 1;
  return buildNamedOp(
      [&]() { return b.create<comb::ExtractOp>(loc, v, lo, width); }, name);
}

// Truncates 'value' to its lower 'width' bits.
Value RTLBuilder::truncate(Value value, unsigned width,
                           std::optional<StringRef> name) {
  return extract(value, 0, width - 1, name);
}

Value RTLBuilder::zext(Value value, unsigned outWidth,
                       std::optional<StringRef> name) {
  unsigned inWidth = value.getType().getIntOrFloatBitWidth();
  assert(inWidth <= outWidth && "zext: input width must be <- output width.");
  if (inWidth == outWidth)
    return value;
  auto c0 = constant(outWidth - inWidth, 0);
  return concat({c0, value}, name);
}

Value RTLBuilder::sext(Value value, unsigned outWidth,
                       std::optional<StringRef> name) {
  return comb::createOrFoldSExt(loc, value, b.getIntegerType(outWidth), b);
}

// Extracts a single bit v[bit].
Value RTLBuilder::bit(Value v, unsigned index, std::optional<StringRef> name) {
  return extract(v, index, index, name);
}

// Creates a hw.array of the given values.
Value RTLBuilder::arrayCreate(ValueRange values,
                              std::optional<StringRef> name) {
  return buildNamedOp(
      [&]() { return b.create<hw::ArrayCreateOp>(loc, values); }, name);
}

// Extract the 'index'th value from the input array.
Value RTLBuilder::arrayGet(Value array, Value index,
                           std::optional<StringRef> name) {
  return buildNamedOp(
      [&]() { return b.create<hw::ArrayGetOp>(loc, array, index); }, name);
}

// Muxes a range of values.
// The select signal is expected to be a decimal value which selects starting
// from the lowest index of value.
Value RTLBuilder::mux(Value index, ValueRange values,
                      std::optional<StringRef> name) {
  if (values.size() == 2)
    return b.create<comb::MuxOp>(loc, index, values[1], values[0]);

  return arrayGet(arrayCreate(values), index, name);
}

// Muxes a range of values. The select signal is expected to be a 1-hot
// encoded value.
Value RTLBuilder::ohMux(Value index, ValueRange inputs) {
  // Confirm the select input can be a one-hot encoding for the inputs.
  unsigned numInputs = inputs.size();
  assert(numInputs == index.getType().getIntOrFloatBitWidth() &&
         "one-hot select can't mux inputs");

  // Start the mux tree with zero value.
  // Todo: clean up when handshake supports i0.
  auto dataType = inputs[0].getType();
  unsigned width =
      isa<NoneType>(dataType) ? 0 : dataType.getIntOrFloatBitWidth();
  Value muxValue = constant(width, 0);

  // Iteratively chain together muxes from the high bit to the low bit.
  for (size_t i = numInputs - 1; i != 0; --i) {
    Value input = inputs[i];
    Value selectBit = bit(index, i);
    muxValue = mux(selectBit, {muxValue, input});
  }

  return muxValue;
}
