# reduce
根据reduce在pytorch项目文件中的搜索结果，reduce的分布是这样的

![reduce搜索结果](https://raw.githubusercontent.com/saisai96/hellow-world/master/%E5%9B%BE%E7%89%87/1566176581.jpg)

并且根据THC中代码和该图片来推测

![教程中图片](https://raw.githubusercontent.com/saisai96/hellow-world/master/%E5%9B%BE%E7%89%87/640.webp)

/ATen/native中的cpu和cuda文件对应着reduce的底层操作

调用关系在此图片中

![函数调用关系](https://raw.githubusercontent.com/saisai96/hellow-world/master/%E5%9B%BE%E7%89%87/981132123.png)

下面解释一下具体的调用关系
因为一致没有编译成功，所以把个人认为的全部相关函数写下来，再具体查看相互的调用。这个过程可能会遗漏部分函数
现在找的的最外层的函数应该是/ATen/native中的ReduceOps.cpp中第635行[std](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/ReduceOps.cpp)
从这里开始再调用std_out:635行->调用std_var_out:640行->调用std_var_stub:534行

而对于std_var_stub这个函数，在
[/ATen/native/cpu/ReduceOpsKernel.cpp第172行](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp)
和
[/ATen/native/cuda/ReduceOpsKernel.cu第171行](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/ReduceOpsKernel.cu)
都对它做出了一些操作。
关于其中使用的函数没有找到说明或者定义，但是我认为它的功能应该是为std_var_stub在cpu和cuda的执行链接了对应的函数

如果按照这个思路，那么接下来函数的调用也就顺理成章

在[/ATen/native/cuda/ReduceOpsKernel.cu](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/ReduceOpsKernel.cu)中std_var_kernel_cuda:48行->调用std_var_kernel_impl:28行和35行->调用[/ATen/native/cuda/Reduce.cuh](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/Reduce.cuh)中的gpu_reduce_kernel:639行->调用launch_reduce_kernel:593行->调用reduce_kernel:196行->调用.run()函数:313行

因为这里面很多都是对线程id的操作也有一些类型检测的地方，这些语句的大致意思能看得懂但是，它具体在后续的过程中起到什么作用我不明白。因为我想如果算子有不同，那么要么是在reduce操作的外面，传参时做了一些预处理，要么reduce的操作有根本的区别，所以我大概找了几个地方

## 1
ReduceOpsKernel.cu中std_var_kernel_impl在调用gpu_reduce_kernel时分了两个分支，一个专门传入了一个half，它的注释也很接近我们的问题，这是我认为最清晰明确的一个解决方式，但是我不确定怎么改。
```
template <typename scalar_t>
void std_var_kernel_impl(TensorIterator& iter, bool unbiased, bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  gpu_reduce_kernel<scalar_t, scalar_t, 2>(iter, WelfordOps<scalar_t, scalar_t, int32_t, float, thrust::tuple<scalar_t, scalar_t>> { unbiased, take_sqrt }, WelfordData<scalar_t, int32_t, float> {});
}

template <>
void std_var_kernel_impl<at::Half>(TensorIterator& iter, bool unbiased, bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  gpu_reduce_kernel<at::Half, at::Half, 2>(iter, WelfordOps<at::Half, float, int32_t, float, thrust::tuple<at::Half, at::Half>> { unbiased, take_sqrt }, WelfordData<float, int32_t, float> {});
}
```
## 2
Reduce.cuh中launch_reduce_kernel被gpu_reduce_kernel在最后调用，它也可能起到之前std_var_kernel_impl调用gpu_reduce_kernel时的作用。
```
template<int nt, typename R>
static void launch_reduce_kernel(const ReduceConfig& config, const R& reduction) {
  dim3 block = config.block();
  dim3 grid = config.grid();

  auto stream = at::cuda::getCurrentCUDAStream();
  int shared_memory = config.shared_memory_size();
  reduce_kernel<nt, R><<<grid, block, shared_memory, stream>>>(reduction);
  AT_CUDA_CHECK(cudaGetLastError());
}
```

## 3
Reduce.cuh中run函数在结构体ReduceOp中，该函数做了许多if去处理不同的reduce，有的是用线程有的是用块有的是用global。并不是和我们一样，全部用同一种线程id的赋值方式
```
  C10_DEVICE void run() const {
    extern __shared__ char shared_memory[];
    index_t output_idx = config.output_idx();
    index_t input_idx = config.input_idx();
    auto base_offsets = output_calc.get(output_idx);

    arg_t value = ident;
    if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
      auto input_slice = (const char*)src + base_offsets[1];
      value = thread_reduce((const scalar_t*)input_slice);
    }
    
    if (config.should_block_y_reduce()) {
      value = block_y_reduce(value, shared_memory);
    }
    if (config.should_block_x_reduce()) {
      value = block_x_reduce(value, shared_memory);
    }

    auto out = (out_scalar_t*)((char*)dst[0] + base_offsets[0]);
    arg_t* acc = nullptr;
    if (acc_buf != nullptr) {
      size_t numerator = sizeof(arg_t);
      size_t denominator = sizeof(out_scalar_t);
      reduce_fraction(numerator, denominator);
      acc = (arg_t*)((char*)acc_buf + (base_offsets[0] * numerator / denominator));
    }

    if (config.should_global_reduce()) {
      value = global_reduce(value, acc, shared_memory);
    } else if (config.should_store(output_idx)) {
      if (acc == nullptr) {
        if (accumulate) {
          value = accumulate_in_output<can_accumulate_in_output>(out, value);
        }
        if (final_output) {
          set_results_to_output(value, base_offsets[0]);
        } else {
          *out = get_accumulated_output<can_accumulate_in_output>(out, value);
        }
      } else {
        if (accumulate) {
          value = ops.combine(*acc, value);
        }
        if (final_output) {
          set_results_to_output(value, base_offsets[0]);
        } else {
          *acc = value;
        }
      }
    }
  }
```

## 4
[SharedReduceOps.h](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/SharedReduceOps.h)这个函数中规定了很多操作的结构体，并且每个结构体中均有reduce，combine，project。这个与我们的应该很相似，只是它将reduce，加和，赋值分成了三个操作分别去做
```
template <typename acc_t>
struct NormOps {
  acc_t norm;

  inline C10_DEVICE acc_t reduce(acc_t acc, acc_t data) const {
    return acc + compat_pow(std::abs(data), norm);
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return compat_pow(a, acc_t(1.0)/norm);
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
#endif

  NormOps(acc_t norm): norm(norm) {
  }
};
```