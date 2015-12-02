#include <stdio.h>
#include <assert.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <sys/time.h>


// Placeholder for longer list of primes
struct list_node{
  unsigned long long value;
  list_node* next;
};
list_node* prime_list;

// List of primes less than 100 to be checked for divisibility
__device__ const unsigned long long small_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
__device__ const int small_primes_size = 25;

// Function prototypes
__device__ bool basic_test(unsigned long long);
__device__ bool exact_test(unsigned long long);
__device__ bool fermat_test(unsigned long long, curandState state);
__device__ bool miller_rabin_test(unsigned long long);


///////////////////////////////////////////////////////////////////////////////
// Kernel functions
///////////////////////////////////////////////////////////////////////////////


// Generate an initial list of numbers to test for primality
// start must be a multiple of 6 for this to be correct
__global__ void primeCandidates(int count, unsigned long long start, unsigned long long* list)
{
  for(int i = count/2*threadIdx.x; i < count/2*(threadIdx.x+1); i++){
    list[2*i] = start + 6*i - 1;
    list[(2*i)+1] = start + 6*i + 1;
  }
}

// Perform basic filters to eliminate obvious composite numbers.
__global__ void filterCandidates(int count, unsigned long long* list){
  for(int i = count*threadIdx.x; i < count*(threadIdx.x+1); i++){
    if (!(basic_test(list[i]))){
      list[i] = 0;
    }
  }
}

// Perform more rigorous tests to confirm a number is prime
__global__ void testCandidates(int count, unsigned long long* list){
  int idx = threadIdx.x;
  curandState state;
  curand_init(idx, idx, 0, &state);
  for(int i = threadIdx.x * count; i < (threadIdx.x + 1)*count; i++){
    int exact = 1;
    if (list[i] == 0) continue;
    if (!exact_test(list[i])) exact = 0;
    if (!fermat_test(list[i], state)){
       if(exact) printf("  %d\n",list[i]);
       list[i] = 0;
    }
    else{
       if(!exact) printf("    %d\n",list[i]);
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
// Device helper functions
///////////////////////////////////////////////////////////////////////////////


// Tests for divisibility against the list of small primes
__device__ bool basic_test(unsigned long long n){
  for(int i = 0; i < small_primes_size; i++){
    if (!(n % small_primes[i])) return false;
  }
  return true;
}

// Exhaustively search possible divisors to confirm a number is prime.
__device__ bool exact_test(unsigned long long n){
  for(unsigned long long i = 101; i * i <= n; i += 2){
    if (!(n % i)) return false;
  }
  return true;
}

// Perform Fermat's primality test for a given number
__device__ bool fermat_test(unsigned long long n, curandState state){
  int k = 10;
  for(int i = 0; i < k; i++){
    double x = curand_uniform_double(&state);
    unsigned long long a = x * (n-4) + 2;
    unsigned long long b = 1;
    unsigned long long e = n-1;
    while(e > 0){
      if (e & 1) b = (b * a) % n;
      e >>= 1;
      a = (a * a) % n;
    }
    if (b != 1) return false;
  } 
  return true;
}

// Perform the Miller-Rabin primality test for a given number
__device__ bool miller_rabin_test(unsigned long long n){
  return false;
}


///////////////////////////////////////////////////////////////////////////////
// Host helpers
///////////////////////////////////////////////////////////////////////////////

// Placeholder for building linked list of primes
void build_primes(unsigned long long start){
  list_node* node;
  cudaMalloc((void**)&node, sizeof(list_node));
  node->value = 2;
  node->next = NULL;
  prime_list = node;
  for(int i = 3; i * i < start; i+= 2){
    
  }
}


///////////////////////////////////////////////////////////////////////////////
// Program main
///////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{

  // Initialization
  const int count = 32;  // Ints to process per thread. Must be even
  const int num_threads = 32;  // Threads to launch in a single 1-D block
  const int list_size = count * num_threads;
  const unsigned long long start = 60000;
  unsigned long long* list;  // Device pointer to potential primes
  cudaMalloc((void**)&list, list_size * sizeof(unsigned long long));
  dim3 gridSize(1,1,1);
  dim3 blockSize(num_threads, 1, 1);
  struct timeval tv;
  struct timezone tz;
  clock_t startTime, endTime, elapsedTime;
  double timeInSeconds;
  long GTODStartTime, GTODEndTime;

  startTime = clock();
  gettimeofday(&tv, &tz);
  GTODStartTime = tv.tv_sec * 1000 + tv.tv_usec / 1000;

  // First, generate a list of prime candidates
  primeCandidates<<<gridSize, blockSize>>>(count, start, list);

  // Second, filter the candidates to quickly eliminate composites
  filterCandidates<<<gridSize, blockSize>>>(count, list);

  // Third, confirm if candidates are actually prime
  testCandidates<<<gridSize, blockSize>>>(count, list);

  // Copy list back and display (for debugging)
  unsigned long long h_list[list_size];
  cudaMemcpy(h_list, list, list_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

  gettimeofday(&tv, &tz);
  GTODEndTime = tv.tv_sec * 1000 + tv.tv_usec / 1000;
  endTime = clock();
  elapsedTime = endTime - startTime;
  timeInSeconds = (elapsedTime / (double)CLOCKS_PER_SEC);
  printf("        GetTimeOfDay Time= %g\n", (double)(GTODEndTime - GTODStartTime) / 1000.0);
  printf("        Clock Time       = %g\n", timeInSeconds);

  int nprimes = 0;
  for(int i = 0; i < list_size; i++){
    if (h_list[i] != 0) {
      //printf("%llu\n",h_list[i]);
      nprimes++;
    }
  }
  printf("Number of primes: %d\n",nprimes);

  return 0;
}


