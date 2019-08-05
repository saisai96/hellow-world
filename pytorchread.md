# Pytorch源文件简介

## 说明

在开始之前，说明一下本文的目的，本文并不是一个详尽的关于Pytorch的介绍文档，在本文中，我们将更加关注于该文件中的c\c++实现部分，尤其是其中关于cuda的相关操作，现在包含了sofmax、hardtanh、reducesum。对于函数说明将按照：前馈-反馈-与NiuTensor比较的顺序进行。
* 本文中使用了vs2017在window平台来分析源码，如何建立工程，将在文档下一张进行说明。
* 此pytorch源码下载自[github](https://github.com/pytorch/pytorch)，因pytorch官方仍在更新，未来的目录结构可能与本文结构有差别，具体更改应该参考[文档](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)。
* 本文参考材料包括前文提到的源码文件以及博客等等，参考资料将会在文章结尾列出。
* 本文第一次编辑来自东北大学自然语言处理实验室-低精度小组-单韦乔-2019.07.29

## 建立项目

## Pytorch源码目录结构

在Pytorch项目文件下，有以下几个文件夹，它们分别是：
* c10 ： （/c10）工作在各个位置包括服务端和移动端的核心库文件，aten/src/ATen/core中的内容正被慢慢移植到此文件夹内
* aten ：（/aten/scr）Pytorch的C++张量库（不支持自动微分），其中包含了ATen、TH、THC、THCUNN、THNN（其中除了ATen和一些.cu文件以外大部分是“传统算子”，使用c语言实现，因c语言没有模板类，所以使用宏定义来代替模板类）
   * TH ： Tensor构造相关函数的cpu实现，例如生成、内存、大小、维度等等
   * THC ： Tensor构造相关函数的gpu实现，含有.cpp、.h、.cu、.cuh
   * THNN ： Tensor相关数学函数库的cpu实现
   * THCUNN ： Tensor相关数学函数库的gpu实现，来自于原始Pytorch的库函数，这个文件夹内的内容正被缓慢移植到aten/src/ATen/native中
   * ATen ：（/aten/src/ATen）
      * core ： ATen中的核函数，正迁移到c10中
      * native ： （/aten/src/ATen/native）一些操作的现代实现，可以在此添加一些新的函数
         * cpu ： 不是操作的真正的CPU执行，而是使用了特殊的处理器指令进行编译的执行，详见[README](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/README.md)
         * cuda ： 操作的CUDA执行，这也是我们所关注的地方
         * sparse ： 稀疏矩阵的CPU和CUDA操作
         * mkl、mkldnn、miopen、cudnn ： 均是一些依赖于后端库的操作的执行
* torch ： （/torch）真正的Pytorch库
   * csrc ： 由c++文件组成的Pytorch库，其中混合了Python的胶水代码
* test ： （/test）Pytorch的Python前端的测试单元
* tools ： Pytorch库的代码生成脚本
* caffe2 ： Caffe2的库文件

## 一些先验知识

* 与先前在目录中提到的一致，Pytorch源码中分为“传统算子”和“原生算子”，“原生算子”是算子的现代C++实现，而“传统算子”则是c实现。所以“传统算子”部分的代码实现与我们所熟悉的结构有些不同，它使用了大量的宏定义来代替模板类的功能（包括一些必要的中间宏定义），而为了节省代码量，它总是要在include包括基本函数的文件时进行重定义。例如：\aten\src\TH\THGenerateDoubleType.h和\aten\src\TH\generic\THTensorLapack.h
* Tensor的属性和成员函数定义在\aten\src\ATen\core\Tensor.h当中
* .cu文件中存在着一些巧妙的类型的特偏化，例如类型一样会匹配下面的SameType，而不同时会匹配上面的
```
template <typename T, typename U>
struct SameType {
  static const bool same = false;
};
```
```
template <typename T>
struct SameType<T, T> {
  static const bool same = true;
};
```
```
SameType<int, int>::same # 这里为true
SameType<int, float>::same #　这里为false
```
* 在c宏定义中Real用来替换、拼接函数名，
* \c10\Half.h（关于半精度类型转换等一些函数）
* \aten\src\THC\THCNumerics.cuh（所有精度的全部数学函数）

## 函数的cuda实现

### SoftMax.cu(\aten\src\ATen\native\cuda)

#### 前馈：

**其中调用顺序是 ：log_softmax_cuda->host_softmax->cunn_SoftMaxForWard**

1. softmax_cuda以及SoftMaxForwardEpilogue：
   * 首先调用softmax_cuda函数， 函数接受一个 Tensor类型的引用， 一个__int64型的维度信息和一个bool类型的转型信息（为真时表示input是half类型，需要进行类型转换）
   * 在此函数中调用host_softmax， 并将上述三参数，以及结构体SoftMaxForwardEpilogue传入。其中结构体的作用是logsoftmax函数本体，传入其的主要作用是方便选择使用logsoftmax函数还是使用softmax函数。

```
Tensor softmax_cuda(const Tensor &input, const int64_t dim, const bool half_to_float){
  return host_softmax<SoftMaxForwardEpilogue>(input, dim, half_to_float);
}
```
```
template<typename T, typename AccumT, typename OutT>
struct LogSoftMaxForwardEpilogue {
  __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
    : logsum(max_input + std::log(sum)) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>(input - logsum);
}

  const AccumT logsum;
};
```
---
2. host_softmax：
   * 在host_softmax中，首先使用half_to_float参数进行类型转换判断（类型转换仅仅支持half，对于int8和int类型应该不支持进行softmax）， 并使用input类型为output的类型赋值
   * 校验dim参数的合法性
   * 当input内数据不为空时，进行维度计算
   * 当inner_size为1时，分配grid和block并调用cunn_SoftMaxForWard（其不为1时应该使用的时二维的grid，此变量应只与线程的分配有关）
```
template<template<typename, typename, typename> class Epilogue>
Tensor host_softmax(const Tensor & input_, const int64_t dim_, const bool half_to_float){
  if (half_to_float) AT_ASSERTM(input_.type().scalarType() == ScalarType::Half,"conversion is supported for Half type only");
  auto input = input_.contiguous();
  Tensor output = half_to_float ? at::empty_like(input, input.options().dtype(ScalarType::Float)) : at::empty_like(input);
  static_assert(std::is_same<acc_type<at::Half, true>, float>::value, "accscalar_t for half should be float");
  if (input.dim() == 0) input = input.view(1);
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  AT_CHECK(dim >=0 && dim < input.dim(), "dim must be non-negative and less than input dimensions");
  int64_t outer_size = 1;
  int64_t dim_size = input.size(dim);

  if (input.numel() > 0) {
    int64_t inner_size = 1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int64_t i = 0; i < dim; ++i)
      outer_size *= input.size(i);
    for (int64_t i = dim + 1; i < input.dim(); ++i)
      inner_size *= input.size(i);
    // This kernel spawns a block per each element in the batch.
    // XXX: it assumes that inner_size == 1
    if (inner_size == 1) {
      const int ILP = 2;
      dim3 grid(outer_size);
      dim3 block = SoftMax_getBlockSize(ILP, dim_size);
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "host_softmax", [&] {
      using accscalar_t = acc_type<scalar_t, true>;
      if (!half_to_float) {
          cunn_SoftMaxForward<ILP, scalar_t, accscalar_t, scalar_t, Epilogue>
            <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
              output.data<scalar_t>(), input.data<scalar_t>(), dim_size
          );
      } else {
          cunn_SoftMaxForward<ILP, scalar_t, accscalar_t, accscalar_t, Epilogue>
            <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
              output.data<accscalar_t>(), input.data<scalar_t>(), dim_size
          );
      }
      });
    // This kernel runs in a 2D grid, where each application along y dimension has a fixed
    // outer_size, and runs in parallel over inner_size. Dimension x is parallel over outer_size.
    // Reductions over dim are done in a single-threaded manner.
    } else {
      uint32_t smem_size;
      dim3 grid, block;
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "host_softmax", [&] {
      using accscalar_t = acc_type<scalar_t, true>;
      if (!half_to_float) {
          SpatialSoftMax_getLaunchSizes<accscalar_t>(
              &cunn_SpatialSoftMaxForward<scalar_t, accscalar_t, scalar_t, Epilogue>,
              outer_size, dim_size, inner_size,
              grid, block, smem_size);
          cunn_SpatialSoftMaxForward<scalar_t, accscalar_t, scalar_t, Epilogue>
            <<<grid, block, smem_size, stream>>>(
             output.data<scalar_t>(), input.data<scalar_t>(), outer_size, dim_size, inner_size
      );
      } else {
          SpatialSoftMax_getLaunchSizes<accscalar_t>(
              &cunn_SpatialSoftMaxForward<scalar_t, accscalar_t, accscalar_t, Epilogue>,
              outer_size, dim_size, inner_size,
              grid, block, smem_size);
          cunn_SpatialSoftMaxForward<scalar_t, accscalar_t, accscalar_t, Epilogue>
            <<<grid, block, smem_size, stream>>>(
             output.data<accscalar_t>(), input.data<scalar_t>(), outer_size, dim_size, inner_size
      );
      }
      });
    }
    THCudaCheck(cudaGetLastError());
  }
  return output;
}
```
---
3. cunn_SoftMaxForward：
   * 在cunn_SoftMaxForward中对指定blockidx的数据进行计算，并分别调用ilpReduce和blockReduce求行的max和exp
   * 根据得到的max和exp初始化epilogue函数，并使用epilogue计算softmax或logsoftmax得到output
```
template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxForward(outscalar_t *output, scalar_t *input, int classes)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += blockIdx.x * classes;
  output += blockIdx.x * classes;

  // find the max
  accscalar_t threadMax = ilpReduce<MaxFloat, ILP, scalar_t, accscalar_t>(
      input, classes, MaxFloat<scalar_t, accscalar_t>(), -at::numeric_limits<accscalar_t>::max());
  accscalar_t max_k = blockReduce<Max, accscalar_t>(
      sdata, threadMax, Max<accscalar_t>(), -at::numeric_limits<accscalar_t>::max());

  // reduce all values
  accscalar_t threadExp = ilpReduce<SumExpFloat, ILP, scalar_t, accscalar_t>(
      input, classes, SumExpFloat<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
  accscalar_t sumAll = blockReduce<Add, accscalar_t>(
      sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));

  Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_k, sumAll);
  int offset = threadIdx.x;
  int last = classes % (ILP * blockDim.x);
  for (; offset < classes - last; offset += blockDim.x * ILP) {
    scalar_t tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      tmp[j] = input[offset + j * blockDim.x];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      output[offset + j * blockDim.x] = epilogue(tmp[j]);
  }
```
---

#### 反馈：

**反馈函数的调用顺序 ： softmax_backward_cuda->host_softmax_backward->cunn_SoftMaxBackward**


*因一下几个函数的代码行为十分类似于前馈，所以接下来就不贴全部代码了，只是简要说明一下有区别的地方*

1. softmax_backward_cuda以及LogSoftMaxBackwardEpilogue：
   * 在反馈函数中，softmax_backward_cuda接受4个参数，3个Tensor类型的张量，分别是梯度、输入与输出，以及一个__int64的维度信息
   * 在其中的的操作相比于前向反馈多了一个关于半精度梯度的判断，保证半精度的梯度必须为全精度
   * 同样，结构体LogSoftMaxBackwardEpilogue也作为参数传入，作为最后计算logsoftmaxbackward的公式
```
Tensor softmax_backward_cuda(const Tensor &grad, const Tensor &output, int64_t dim, const Tensor &input){
  bool half_to_float = grad.type().scalarType() != input.type().scalarType();
  if (half_to_float) {
     AT_ASSERTM((grad.type().scalarType() == ScalarType::Float && input.type().scalarType() == ScalarType::Half), "expected input and grad types to match, or input to be at::Half and grad to be at::Float");
  }
  Tensor tmp = grad * output;
  return host_softmax_backward<SoftMaxBackwardEpilogue>(tmp, output, dim, half_to_float);
}
```
```
template<typename T, typename AccumT, typename OutT>
struct LogSoftMaxBackwardEpilogue {
  __device__ __forceinline__ LogSoftMaxBackwardEpilogue(AccumT sum)
    : sum(sum) {}

  __device__ __forceinline__ T operator()(OutT gradOutput, OutT output) const {
    return static_cast<T>(gradOutput - std::exp(static_cast<AccumT>(output)) * sum);
  }

  const AccumT sum;
};
```
---
2. cunn_SoftMaxBackward：
   * 在cunn_SoftMaxBackward中因传入的epilogue不同，所以最后的数值计算方式采用反馈的计算方式，具体的计算过程可以参考前文中的LogSoftMaxForwardEpilogue以及LogSoftMaxBackwardEpilogue
---
#### 相比于NiuTensor

1. softmax_cuda/log_softmax_backward_cuda： 
函数可以相当于NiuTensor中函数的.cpp文件，两个函数均是一个入口，但这样的函数放入.cpp中用作用户直接调用的话还是有些复杂。然而单独写出个人感觉也没有什么必要，因为其参数、返回值等等与其调用的host_softmax/host_softmax_backward没有区别，函数中也没有做特殊的操作。
2. host_softmax/host_softmax_backward： 
相当于NiuTensor中的_Cuda***函数，作为global函数供主机调用，并以此连接设备函数，此函数中也是做一些计算的准备工作，包括Tensor类型，是否为空，线程维度等等的判断与初始化
3. cunn_SoftMaxForward/cunn_SoftMaxBackward： 
这两个函数相当于NiuTensor中的Kernel***函数，但它还是一个global函数可以在.cpp文件中直接被调用，在这两个函数中主要做了一些申请内存，线程索引赋值，定义中间变量的工作，真正的计算工作被交给了ilpReduce、blockReduce以及LogSoftMaxForwardEpilogue等带有Epilogue后缀的函数完成。

---
### HardTanh.cu(\aten\src\THCUNN\generic)

#### 前馈：

* THNN_:
   * 因hardtanh函数本身只是做了一个简单的函数映射，代码实现的思路也很好懂，没什么好说的
```
void THNN_(HardTanh_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accreal min_val_,
           accreal max_val_,
           bool inplace)
{
  scalar_t min_val = ScalarConvert<accreal, scalar_t>::to(min_val_);
  scalar_t max_val = ScalarConvert<accreal, scalar_t>::to(max_val_);

  THCUNN_assertSameGPU(state, 2, input, output);
  if(inplace)
  {
    THCTensor_(set)(state, output, input);
    THC_pointwiseApply1<scalar_t>(state, output, hardtanhupdateOutput_functor<scalar_t>(min_val, max_val));
  }
  else
  {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply2<scalar_t, scalar_t>(state, output, input,
                               hardtanhupdateOutput_functor<scalar_t>(min_val, max_val));
  }
}
```
```
template <typename T>
struct hardtanhupdateOutput_functor
{
  const T max_val_;
  const T min_val_;

  hardtanhupdateOutput_functor(T min_val, T max_val)
      : max_val_(max_val), min_val_(min_val) {}

  __device__ void operator()(T *output, const T *input) const
  {
    if (*input < min_val_)
      *output = min_val_;
    else if (*input > max_val_)
      *output = max_val_;
    else
      *output = *input;
  }

  __device__ void operator()(T *input) const
  {
    if (*input < min_val_)
      *input = min_val_;
    else if (*input > max_val_)
      *input = max_val_;
  }
};
```
---
#### 反馈：

* 与前馈类似

---
#### 相比于NiuTensor
* 几乎没有区别

---
### THCTensorMathReduce.cu(\aten\src\THC)

**这个文件中的函数作为入口，调用了其他函数，实现了reducesum函数的功能**
**函数的调用顺序是THCudaByteTensor_logicalAnd->THC_reduceDim->kernelReduceContigDim->reduceBlock->reduceNValuesInBlock**

*因项目内该位置没有注释，同时也没有其他函数调用它，所以传入的参数意义还不是很清楚，只能从类型和参数名做大致推断*

1. THCudaByteTensor_logicalAnd：
   * 根据参考资料，这个函数应有与对应的cpu版本，但是还未找到
   * 具体的函数功能是通过调用THC_reduceDim实现的， 这里注意到其传入了模板类类型uint8_t。因源文件除了直接调用的函数所在的文件被include以外，其他文件均未明确包含在内，但根据uint8_t的搜索结果，它在文件中应表示的格式是__int8或unsigned char，所以推测，这个入口函数应该是只处理__int8，reducesum的全精度以及半精度实现应在其他文件夹内。
```
THC_API void
THCudaByteTensor_logicalAnd(THCState* state, THCudaByteTensor *self, THCudaByteTensor *src, int dimension, int keepdim) {
  THCAssertSameGPU(THCudaByteTensor_checkGPU(state, 2, self, src));
  if (!THC_reduceDim<uint8_t>(state, self, src,
                              thrust::identity<unsigned char>(),
                              LogicalAll(),
                              thrust::identity<unsigned char>(),
                              (unsigned char) 1,
                              dimension,
                              keepdim)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}
```
---
2. THC_reduceDim：
   * 在函数的开始进行了一些维度赋值，参数合法性的判断，其中赋值的参数主要用于后面的grid等索引的计算
   * 然后做一个判断（Is the reduction dimension contiguous?）是否是连续内存，并根据判断结果采用不同的grid、block赋值方式
   * 最后调用kernelReduceContigDim，计算reducesum（对于不连续的部分有单独的函数处理，这种方式提高了函数的处理速度）
```
// Performs a reduction out[..., 0, ...] = reduce_i(modify(in[..., i, ...])) for
// all in where i and the out's 0 are indexed at dimension `dim`
template <typename ScalarType,
typename TensorType,
typename ModifyOp,
typename ReduceOp,
typename FinalizeOp,
typename AccT>
bool THC_reduceDim(THCState* state,
                   TensorType* out,
                   TensorType* in,
                   const ModifyOp modifyOp,
                   const ReduceOp reduceOp,
                   const FinalizeOp finalizeOp,
                   AccT init,
                   int dim,
                   int keepdim) {
  ptrdiff_t inElements = THCTensor_nElement(state, in);//元素个数

  int64_t reductionSize = THTensor_sizeLegacyNoScalars(in, dim);
  int64_t reductionStride = THTensor_strideLegacyNoScalars(in, dim);
  ptrdiff_t outElements = inElements / reductionSize;

  if (THCTensor_nDimensionLegacyAll(state, out) > MAX_CUTORCH_DIMS ||
      THCTensor_nDimensionLegacyAll(state, in) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (THCTensor_nDimensionLegacyAll(state, in) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  // Is the reduction dimension contiguous? If so, then we can use a
  // shared memory reduction kernel to increase performance.
  bool contigReduction = (reductionStride == 1);

  dim3 block;
  dim3 grid;
  int smemSize = 0; // contiguous reduction uses smem
  if (contigReduction) {
    if (!getContigReduceGrid(outElements, grid)) {
      return false;
    }

    block = getContigReduceBlock(outElements, reductionSize);
    smemSize = sizeof(AccT) * block.x;
  } else {
    if (!getNoncontigReduceGrid(outElements, grid)) {
      return false;
    }

    block = getNoncontigReduceBlock();

    if(outElements <= 4096)
    {
      // gridDim.x and blockDim.x parallelize work across slices.
      // blockDim.y enables some intra-block reduction within slices.
      // gridDim.y enables inter-block reduction within slices.

      // Each block covers 32 output elements.
      int blockdimx = 32;
      int griddimx = THCCeilDiv((int64_t)outElements, (int64_t)blockdimx);

      // Each warp reduces at most 4 slices.  This heuristic can be tuned,
      // but locking blockdimy to 16 is robust and reasonably performant.
      int blockdimy = 16;

      int griddimy = 1;
      bool coop = false;
      // Rough heuristics to decide if using cooperating blocks is worthwhile
      if(                      outElements <=   32 && reductionSize >= 4096) coop = true;
      if(  32 < outElements && outElements <=   64 && reductionSize >= 4096) coop = true;
      if(  64 < outElements && outElements <=  128 && reductionSize >= 4096) coop = true;
      if( 128 < outElements && outElements <=  256 && reductionSize >= 4096) coop = true;
      if( 256 < outElements && outElements <=  512 && reductionSize >= 4096) coop = true;
      if( 512 < outElements && outElements <= 1024 && reductionSize >= 4096) coop = true;
      if(1024 < outElements && outElements <= 2048 && reductionSize >= 2048) coop = true;
      if(2048 < outElements && outElements <= 4096 && reductionSize >= 2048) coop = true;
      // Each block reduces at most CHUNKPERBLOCK (currently 256) slices.
      if(coop)
        griddimy = THCCeilDiv((int64_t)reductionSize, (int64_t)CHUNKPERBLOCK);//计算grid

      grid = dim3(griddimx, griddimy, 1);
      block = dim3(blockdimx, blockdimy, 1);
    }
  }

  // Resize out to correspond to the reduced size with keepdim=True.

  // Preserve noncontiguities by unsqueezing out if necessary
  THCTensor_preserveReduceDimSemantics(
      state, out, THCTensor_nDimensionLegacyAll(state, in), dim, keepdim);

  // Resize out
  std::vector<int64_t> sizes = THTensor_sizesLegacyNoScalars(in);
  sizes[dim] = 1;
  THCTensor_resize(state, out, sizes, {});

  // It is possible that the tensor dimensions are able to be collapsed,//tensor的维度可以被压缩
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, OUT, IN)                                      \
  if (contigReduction) {                                                \
    kernelReduceContigDim<ScalarType,                                   \
                          TYPE, AccT, ModifyOp, ReduceOp, FinalizeOp,   \
                          OUT, IN>                                      \
      <<<grid, block, smemSize, THCState_getCurrentStream(state)>>>     \
        (outInfo, inInfo, reductionSize,                                \
        (TYPE) outElements, init, modifyOp, reduceOp, finalizeOp);      \
  } else {                                                              \
    if(block.y == 1){                                                   \
        kernelReduceNoncontigDim<                                       \
                          ScalarType,                                   \
                          TYPE, AccT, ModifyOp, ReduceOp, FinalizeOp,   \
                          OUT, IN>                                      \
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>          \
        (outInfo, inInfo, reductionStride, reductionSize,               \
        (TYPE) outElements, init, modifyOp, reduceOp, finalizeOp);      \
    }                                                                   \
    else                                                                \
    {                                                                   \
        void* stagingData = nullptr;                                    \
        void* semaphores = nullptr;                                     \
                                                                             \
        if(grid.y > 1)                                                       \
        {                                                                    \
          stagingData = THCudaMalloc(state, sizeof(AccT)*outElements*grid.y);\
          semaphores = THCudaMalloc(state, sizeof(int)*grid.x);              \
          THCudaCheck(cudaMemsetAsync                                        \
            (semaphores,                                                     \
             0,                                                              \
             sizeof(int)*grid.x,                                             \
             THCState_getCurrentStream(state)));                             \
        }                                                                    \
                                                                             \
        kernelReduceNoncontigDim_shared                                      \
          <ScalarType, TYPE, AccT, ModifyOp, ReduceOp, FinalizeOp,  OUT, IN> \
          <<<grid, block, 0, THCState_getCurrentStream(state)>>>             \
          (outInfo,                                                          \
           inInfo,                                                           \
           reductionStride,                                                  \
           reductionSize,                                                    \
           (TYPE) outElements,                                               \
           init,                                                             \
           modifyOp,                                                         \
           reduceOp,                                                         \
           finalizeOp,                                                       \
           (volatile AccT*)stagingData,                                      \
           (int*)semaphores);                                                \
                                                                             \
        if(grid.y > 1)                                                       \
        {                                                                    \
          THCudaFree(state, stagingData);                                    \
          THCudaFree(state, semaphores);                                     \
        }                                                                    \
    }                                                                        \
  }

#define HANDLE_IN_CASE(TYPE, OUT, IN)                     \
  {                                                       \
    switch (IN) {                                         \
      case 1:                                             \
        HANDLE_CASE(TYPE, OUT, 1);                        \
        break;                                            \
      case 2:                                             \
        HANDLE_CASE(TYPE, OUT, 2);                        \
        break;                                            \
      default:                                            \
        HANDLE_CASE(TYPE, OUT, -1);                       \
        break;                                            \
    }                                                     \
  }

#define HANDLE_OUT_CASE(TYPE, OUT, IN)                    \
  {                                                       \
    switch (OUT) {                                        \
      case 1:                                             \
        HANDLE_IN_CASE(TYPE, 1, IN);                      \
        break;                                            \
      case 2:                                             \
        HANDLE_IN_CASE(TYPE, 2, IN);                      \
        break;                                            \
      default:                                            \
        HANDLE_IN_CASE(TYPE, -1, IN);                     \
        break;                                            \
    }                                                     \
  }

  if(THCTensor_canUse32BitIndexMath(state, out) &&
     THCTensor_canUse32BitIndexMath(state, in))
  {
    TensorInfo<ScalarType,
               unsigned int> outInfo =
      getTensorInfo<ScalarType, TensorType, unsigned int>(state, out);
    outInfo.collapseDims();

    TensorInfo<ScalarType,
               unsigned int> inInfo =
      getTensorInfo<ScalarType, TensorType, unsigned int>(state, in);
    inInfo.reduceDim(dim);
    inInfo.collapseDims();
    HANDLE_OUT_CASE(unsigned int, outInfo.dims, inInfo.dims);
  }
  else
  {
    TensorInfo<ScalarType,
               uint64_t> outInfo =
      getTensorInfo<ScalarType, TensorType, uint64_t>(state, out);
    outInfo.collapseDims();

    TensorInfo<ScalarType,
               uint64_t> inInfo =
      getTensorInfo<ScalarType, TensorType, uint64_t>(state, in);
    inInfo.reduceDim(dim);
    inInfo.collapseDims();

    /*
    Only instantiates the all 1D special case and the fallback all nD case for
    large (64-bit indexed) tensors to reduce compilation time.
    */
    if (outInfo.dims == 1 && inInfo.dims == 1) {
      HANDLE_CASE(uint64_t, 1, 1);
    } else {
      HANDLE_CASE(uint64_t, -1, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_IN_CASE
#undef HANDLE_OUT_CASE


  if (!keepdim) {
    THCTensor_squeeze1d(state, out, out, dim);
  }
  return true;
}
```
---
3. kernelReduceContigDim：
   * 函数最开始计算并行索引，接着开辟共享内存并计算，最后将计算结果赋值到out中
   * 因除了上述第一个函数调用了THC_reduceDim，所以暂时不知道ReduceOp是一个什么样的参数。根据注释等方面的信息，它也做了reduce的工作，但是实现细节还不是很理解
```
// Kernel that handles an entire reduction of a slice of a tensor per
// each block
template <typename T,
          typename IndexType,
          typename AccT,
          typename ModifyOp,
          typename ReduceOp,
          typename FinalizeOp,
          int ADims, int BDims>
__global__ void
kernelReduceContigDim(TensorInfo<T, IndexType> out,
                      TensorInfo<T, IndexType> in,
                      IndexType reductionSize,
                      IndexType totalSlices,
                      AccT init,
                      ModifyOp modifyOp,
                      ReduceOp reduceOp,
                      FinalizeOp finalizeOp) {
  const IndexType sliceIndex = getReduceContigDimSliceIndex<IndexType>();

  if (sliceIndex >= totalSlices) {
    return;
  }

  // Get the offset in `out` for the reduction
  const IndexType outOffset =
    IndexToOffset<T, IndexType, ADims>::get(sliceIndex, out);

  // Get the base offset in `in` for this block's reduction
  const IndexType inBaseOffset =
    IndexToOffset<T, IndexType, BDims>::get(sliceIndex, in);

  // Each thread in the block will reduce some subset of elements in
  // the slice. The elements are guaranteed contiguous starting at
  // `inBaseOffset`.
  AccT r = init;
  for (IndexType i = threadIdx.x; i < reductionSize; i += blockDim.x) {
    const AccT val = scalar_cast<AccT>(in.data[inBaseOffset + i]);
    r = reduceOp(r, modifyOp(val));
  }

  // Reduce within the block
  // FIXME: extern name
  extern __shared__ char smemChar[];
  AccT* smem = (AccT*) smemChar;
  r = reduceBlock<AccT, ReduceOp>(smem, blockDim.x, r, reduceOp, init);

  if (threadIdx.x == 0) {
    // Write out reduced value
    out.data[outOffset] = scalar_cast<T>(finalizeOp(r));
  }
}
```
---
4. reduceBlock以及reduceNValuesInBlock：
   * reduceBlock调用reduceNValuesInBlock来进行reducesum的运算
   * reduceNValuesInBlock中reduceOp是关键计算步骤
```
template <typename T, typename ReduceOp>
__device__ T reduceBlock(T* smem,
                         const unsigned int numVals,
                         T threadVal,
                         ReduceOp reduceOp,
                         T init) {
  reduceNValuesInBlock<T, ReduceOp, 1>(smem, &threadVal, numVals, reduceOp, init);
  return threadVal;
}
```
```
// Reduce N values concurrently, i.e. suppose N = 2, and there are 4 threads:
// (1, 2), (3, 4), (5, 6), (7, 8), then the return in threadVals for thread 0
// is (1 + 3 + 5 + 7, 2 + 4 + 6 + 8) = (16, 20)
//
// If smem is not used again, there is no need to __syncthreads before this
// call. However, if smem will be used, e.g., this function is called in a loop,
// then __syncthreads is needed either before or afterwards to prevent non-0
// threads overriding smem in the next loop before num-0 thread reads from it.
template <typename T, typename ReduceOp, int N>
__device__ void reduceNValuesInBlock(T *smem,
                             T threadVals[N],
                             const unsigned int numVals,
                             ReduceOp reduceOp,
                             T init) {
  if (numVals == 0) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      threadVals[i] = init;
    }
    return;
  }

  // We store each of the N values contiguously, so if N = 2, all values for
  // the first threadVal for each thread in the block are stored followed by
  // all of the values for the second threadVal for each thread in the block
  if (threadIdx.x < numVals) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      smem[i * numVals + threadIdx.x] = threadVals[i];
    }
  }
  __syncthreads();

  // Number of lanes in the final reduction --> this is used to determine
  // where to put the outputs of each of the n things we are reducing. If
  // nLP = 32, then we have the 32 outputs for the first threadVal,
  // followed by the 32 outputs for the second threadVal, etc.
  const unsigned int numLanesParticipating = min(numVals, warpSize);

  if (numVals > warpSize && ((threadIdx.x / warpSize) == 0 )) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      threadVals[i] = threadIdx.x < numVals ? threadVals[i] : init;
    }

    for (int i = warpSize + threadIdx.x; i < numVals; i += warpSize) {
      #pragma unroll
      for (int j = 0; j < N; ++j) {
        threadVals[j] = reduceOp(threadVals[j], smem[j * numVals + i]);
      }
    }

    #pragma unroll
    for (int i = 0; i < N; ++i) {
      smem[i * numLanesParticipating + threadIdx.x] = threadVals[i];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (numLanesParticipating == 32) {
      #pragma unroll
      for (int i = 0; i < N; ++i) {
        #pragma unroll
        for (int j = 1; j < 32; ++j) {
          threadVals[i] = reduceOp(threadVals[i], smem[i * 32 + j]);
        }
      }
    } else {
      #pragma unroll
      for (int i = 0; i < N; ++i) {
        for (int j = 1; j < numLanesParticipating; ++j) {
          threadVals[i] = reduceOp(threadVals[i], smem[i * numVals + j]);
        }
      }
    }
  }
}
```

## 参考文件

[万字综述，核心开发者全面解读PyTorch内部机制](https://mp.weixin.qq.com/s/8J-vsOukt7xwWQFtwnSnWw)
[PyTorch源码浅析（目录）](https://zhuanlan.zhihu.com/p/34629243)
[PyTorch学习笔记(5)——论一个torch.Tensor是如何构建完成的？](https://blog.csdn.net/g11d111/article/details/81231292)
