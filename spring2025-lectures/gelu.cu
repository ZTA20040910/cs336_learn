#include <c10/cuda/CUDAException.h>
#include <math.h>
#include <torch/extension.h>
// _global__ 说明这个函数是跑在 GPU 上的核函数
__global__ void gelu_kernel(float *in, float *out, int num_elements) {
  // Get the index into the tensor
  // 坐标计算
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // 边界检查。比如有 1005 个数，开了 2 个 block (共2048线程)，
  // 剩下的最后1043个线程发现索引 i 越界了，就直接放工（不执行操作）。
  if (i < num_elements) { // To handle the case when n < numBlocks * blockDim
    // Do the actual computation
    out[i] =
        0.5 * in[i] *
        (1.0 + tanh(0.79788456 * (in[i] + 0.044715 * in[i] * in[i] * in[i])));
  }
}

// 计算向上取整的除法，等同于 Python 里的 math.ceil(a / b)
// 比如 1005 个元素，block 大小 1024，结果就是 1 个 block
inline unsigned int cdiv(unsigned int a, unsigned int b) {
  // Compute ceil(a / b)
  return (a + b - 1) / b;
}

// 接收一个 PyTorch Tensor，返回一个 Tensor
torch::Tensor gelu(torch::Tensor x) {
  // 参数校验，强制张量必须在 GPU 上，且在内存中必须是连续排布的
  TORCH_CHECK(x.device().is_cuda());
  TORCH_CHECK(x.is_contiguous());

  // Allocate empty tensor，准备好存放结果的篮子
  torch::Tensor y = torch::empty_like(x);

  // Determine grid (elements divided into blocks)，规划 GPU 任务
  int num_elements = x.numel(); // 总元素个数
  int block_size = 1024;        // Number of threads，设定每栋楼住 1024 个“工人”
  int num_blocks = cdiv(num_elements, block_size); // 计算需要多少栋“楼”

  // Launch the kernel，启动核函数
  // <<<num_blocks, block_size>>> 是 CUDA 特有的三尖括号语法：
  // 告诉 GPU：我们要开启 num_blocks 个块，每个块 block_size 个线程。
  // x.data_ptr<float>() 获取张量在显存里的原始首地址指针。
  gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(),
                                          y.data_ptr<float>(), num_elements);
  // 错误检查，如果显存溢出了、或者核函数写错了，这一行会立刻抛出 C++
  // 异常，Python 端就能捕获到。
  C10_CUDA_KERNEL_LAUNCH_CHECK(); // Catch errors immediately

  return y;
}