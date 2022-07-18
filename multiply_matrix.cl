__kernel void multiplyMatrices(
                      const __global float* A,
                      const __global float* B,
                      __global float* C,const int M, const int N, const int K) {


    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    float acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += A[k*M + globalRow] * B[globalCol*K + k];
    }
    C[globalCol*M + globalRow] = acc;
}

__kernel void multiplyByColum(
                      const __global float* A,
                      const __global float* B,
                      __global float* C,const int M, const int N, const int K) {


    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    float acc = 0.0f;
    for (int k=0; k<M; k++) {
        acc += A[globalRow*M+k] * B[globalCol*K + k];
    }
    C[globalCol*M + globalRow] = acc;
}



__kernel void sumArrays(__global float* a, __global float* b, __global float* c){
     int index = get_global_id(0);
     c[index] = a[index] + b[index];
 }


 __kernel void subtractArrays(__global float* a, __global float* b, __global float* c){
      int index = get_global_id(0);
      c[index] = a[index] - b[index];
  }


__kernel void multiplyByConst(__global float* a,__global float* c, const float M){
     int index = get_global_id(0);
     c[index] = a[index] * M;
 }

__kernel void addConst(__global float* a,__global float* c, const float M){
     int index = get_global_id(0);
     c[index] = a[index] + M;
 }