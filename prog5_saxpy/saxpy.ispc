
export void saxpy_ispc(uniform int N,
                       uniform float scale,
                       uniform float X[],
                       uniform float Y[],
                       uniform float result[])
{
    foreach (i = 0 ... N) {           
        result[i] = scale * X[i] + Y[i];
    }
}

export void saxpy_streaming_ispc(uniform int N,
                                 uniform float scale,
                                 uniform float X[],
                                 uniform float Y[],
                                 uniform float result[])
{
    // main打印的结果显示:
    // [saxpy ispc]:           [12.506] ms     [23.831] GB/s   [3.199] GFLOPS
    // [saxpy task ispc]:      [17.974] ms     [16.581] GB/s   [2.225] GFLOPS
    // 计算强度仅有1/8左右 (2FLOPs / 16B)
    // 这是因为在写result[i]时默认读后写回, Read For Ownership(RFO)
    // 然而我们不关心读的内容, 多余的读操作不仅挤占带宽还污染缓存
    // 方法: 使用流式读写(非时序读写)绕过缓存, 注意对齐
    // https://ispc.godbolt.org/z/m43bw9

    const uniform intptr_t simdBytes   = sizeof(varying float);
    const uniform intptr_t ptrBytes    = sizeof(uniform float);
    const uniform intptr_t alignMask   = ~(simdBytes - 1);
    const uniform intptr_t rawAddr     = (uniform intptr_t)result;
    const uniform intptr_t alignedAddr = (rawAddr + simdBytes - 1) & alignMask;
    const uniform int      prefixCnt   = min((alignedAddr - rawAddr) / ptrBytes, (uniform int64) N);

    foreach (i = 0 ... prefixCnt) {
        result[i] = scale * X[i] + Y[i];
    }

    uniform int i = prefixCnt;
    for (; i < N; i += programCount) {
        streaming_store(result + i, scale * X[i + programIndex] + Y[i + programIndex]);
    }

    if (i > N) {
        foreach (j = i - programCount ... N) {
            result[i] = scale * X[i] + Y[i];
        }
    }
}

task void saxpy_ispc_task(uniform int N,
                          uniform int span,
                          uniform float scale,
                          uniform float X[], 
                          uniform float Y[],
                          uniform float result[])
{

    uniform int indexStart = taskIndex * span;
    uniform int indexEnd = min(N, indexStart + span);

    foreach (i = indexStart ... indexEnd) {
        result[i] = scale * X[i] + Y[i];
    }
}

export void saxpy_ispc_withtasks(uniform int N,
                               uniform float scale,
                               uniform float X[],
                               uniform float Y[],
                               uniform float result[])
{

    uniform int span = N / 64;  // 64 tasks

    launch[N/span] saxpy_ispc_task(N, span, scale, X, Y, result);
}
