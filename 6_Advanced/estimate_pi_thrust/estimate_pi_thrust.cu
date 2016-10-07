// ref: https://books.google.com/books?id=ODTaCgAAQBAJ&pg=PT562&lpg=PT562&dq=estimate+pi+with+cuda+thrust&source=bl&ots=IDMmU3Ld7Y&sig=dGojekciyDV6f7OoKwijXUZvSXk&hl=en&sa=X&authuser=0#v=onepage&q=estimate%20pi%20with%20cuda%20thrust&f=false
// https://github.com/ishanthilina/CUDA-Calculation-Experiements/blob/master/q1/pi-curand-thrust.cu

#include <iostream>
#include <sstream>
#include <cmath>
#include <sys/time.h>

// #include <thread>

#include <omp.h>

#include <thrust/transform_reduce.h>
#include <thrust/random.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/cpp/execution_policy.h>


//#define UNUSED(x) (void)x;
#define UNUSED(x) [&x]{}()


typedef long long unsigned int UINT64;

int main(int argc, char* argv[]) {
    struct timeval t1, t2;

    auto print_msg = [&] (const std::string & device, float pi) {
        double tdelta =
            (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec)
            / 1000000.0; // in s
        std::cout << "\n" << device << " Est. pi = " << pi
            << "\nTime: " << tdelta << " s"
            << std::endl;
    };

    //int N = (1 << 31);
    UINT64 N = 10000000; // = 0xFFFFFFFFF;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // N = prop.totalGlobalMem * 0.75;

    if (argc > 1) {
        N = atof(argv[1]);
    }
    std::cout << "Trials: " << N << std::endl;

    srand(time(NULL));

    // generate pi trial
    auto genpit = [=] __host__ __device__ (UINT64 thread_id) {
        float x, y;
        #ifdef __CUDA_ARCH__ // runtime macro
            thrust::default_random_engine rng(clock());
            thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
            rng.discard(thread_id);
            x = dist(rng);
            y = dist(rng);
        #else
            x = (float)rand()/RAND_MAX;
            y = (float)rand()/RAND_MAX;
        #endif
        return (x * x + y * y) <= 1.0f;
    };

    std::cout << "calculate pi \n";

    gettimeofday(&t1, 0);
    float pi_count = thrust::transform_reduce(thrust::cuda::par,
        thrust::counting_iterator<UINT64>(0),
        thrust::counting_iterator<UINT64>(N),
        genpit,
        0.0f, thrust::plus<float>());
    cudaDeviceSynchronize();
    gettimeofday(&t2, 0);
    // std::cout << "pi count: " << pi_count << std::endl;
    float pi = pi_count * 4.f / (float) N;
    print_msg("GPU", pi);

    gettimeofday(&t1, 0);
    pi_count = thrust::transform_reduce(thrust::cpp::par,
        thrust::counting_iterator<UINT64>(0),
        thrust::counting_iterator<UINT64>(N),
        genpit,
        0.0f, thrust::plus<float>());
    gettimeofday(&t2, 0);
    //std::cout << "CPP pi count: " << pi_count << std::endl;
    pi = pi_count * 4.f / (float) N;
    print_msg("CPP", pi);

    //unsigned int total_threads = std::thread::hardware_concurrency();
    //std::cout << "\nTotal threads " << total_threads << std::endl;

    // The thrust::omp::par profile very slow. Implementing it manually.
    int nthreads;
    pi_count=0;
    gettimeofday(&t1, 0);
    #pragma omp parallel reduction(+:pi_count)
    {
        nthreads = omp_get_num_threads();
        // int chunk = N / (UINT64) nthreads + 1; //100000;
        // UNUSED(chunk); // inhibit warning

        //#pragma omp for schedule(dynamic,chunk)
        #pragma omp for schedule(static)
        for (UINT64 i = 0; i < N; i++) {
            pi_count += genpit(i);
        }
    }
    gettimeofday(&t2, 0);
    pi = pi_count * 4.f / (float) N;

    std::stringstream msg;
    msg.str(std::string());
    msg << "OMP using " << nthreads << " threads.";
    print_msg(msg.str(), pi);

    return 0;
}
