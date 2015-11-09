#include <stdio.h>
#include <assert.h>

// List of primes less than 100 to be checked for divisibility
__device__ const int small_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
__device__ const int small_primes_size = 25;

// Function prototypes
__device__ bool basic_test(unsigned long long);

// Generate an initial list of numbers to test for primality
__global__ void primeCandidates(int count, unsigned long long* list)
{
  for(int i = count/2*threadIdx.x; i < count/2*(threadIdx.x+1); i++){
    list[2*i] = 6*i - 1;
    list[(2*i)+1] = 6*i + 1;
  }
}

// Perform basic filters to eliminate obvious composite numbers
__global__ void filterCandidates(int count, unsigned long long* list){
  for(int i = count*threadIdx.x; i < count*(threadIdx.x+1); i++){
    if (!(basic_test(list[i]))){
      list[i] = 0;
    }
  }
}

// Perform more rigorous tests to confirm a number is prime
__global__ void testCandidates(unsigned long long* list){

}

// Tests for divisibility against the list of small primes
__device__ bool basic_test(unsigned long long n){
  for(int i = 0; i < small_primes_size; i++){
    if (!(n % small_primes[i])) return false;
  }
  return true;
}

// Perform Fermat's primality test for a given number
__device__ bool fermat_test(unsigned long long n){
  return false;
}

// Perform the Miller-Rabin primality test for a given number
__device__ bool miller_rabin_test(unsigned long long n){
  return false;
}

// Program main
int main( int argc, char** argv) 
{
  // Initialization
  const int count = 100;  // Ints to process per thread
  const int num_threads = 32;  // Threads to launch in a single 1-D block
  const int list_size = count * num_threads;
  unsigned long long* list;  // Device pointer to potential primes
  cudaMalloc((void**)&list, list_size * sizeof(unsigned long long));
  dim3 gridSize(1,1,1);
  dim3 blockSize(num_threads, 1, 1);
  
  // First, generate a list of prime candidates
  primeCandidates<<<gridSize, blockSize>>>(count, list);

  // Second, filter the candidates to quickly eliminate composites
  filterCandidates<<<gridSize, blockSize>>>(count, list);

  // Copy list back and display (for debugging)
  unsigned long long h_list[list_size];
  cudaMemcpy(h_list, list, list_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  for(int i = 0; i < list_size; i++){
    if (h_list[i] != 0) {
      printf("%llu\n",h_list[i]);
    }
  }
  
  return 0;
}

