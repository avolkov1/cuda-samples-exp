// ref: https://books.google.com/books?id=ODTaCgAAQBAJ&pg=PT562&lpg=PT562&dq=estimate+pi+with+cuda+thrust&source=bl&ots=IDMmU3Ld7Y&sig=dGojekciyDV6f7OoKwijXUZvSXk&hl=en&sa=X&authuser=0#v=onepage&q=estimate%20pi%20with%20cuda%20thrust&f=false
// https://github.com/ishanthilina/CUDA-Calculation-Experiements/blob/master/q1/pi-curand-thrust.cu

#include <iostream>
#include <sstream>
#include <cmath>

//#include <chrono>
//#include <thread>

#include <sys/time.h>

#include <omp.h>

#include <thrust/transform_reduce.h>
#include <thrust/random.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/cpp/execution_policy.h>


//#define UNUSED(x) (void)x;
#define UNUSED(x) [&x]{}()

typedef long long unsigned int UINT64;


// Generate pi trial.
//struct Genpit {
//private:
//    thrust::default_random_engine rng_;
//    thrust::uniform_real_distribution<float> dist_;
//
//public:
//    Genpit() {
//        thrust::default_random_engine rng(clock());
//        thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
//        rng_ = rng;
//        dist_ = dist;
//    }
//
//    __host__ __device__ __forceinline__ float operator()(UINT64 id_or_seed)
//    {
//        float x, y;
//        #ifdef __CUDA_ARCH__ // macro true on device and false on host.
//            // thrust::default_random_engine rng(clock());
//            // thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
//            rng_.discard(id_or_seed);
//            x = dist_(rng_);
//            y = dist_(rng_);
//        #else
//            unsigned int rseed_ = id_or_seed;
//            x = (float)rand_r(&rseed_)/RAND_MAX;
//            y = (float)rand_r(&rseed_)/RAND_MAX;
//        #endif
//        return (x * x + y * y) <= 1.0f;
//    }
//};


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

    unsigned int rseed = time(NULL);
    //unsigned int rseed =
    //    static_cast<uint64_t>(std::chrono::system_clock::to_time_t(
    //        std::chrono::system_clock::now()));

    // generate pi trial lambda function.
    auto genpit = [=] __host__ __device__ (UINT64 id_or_seed) {
        float x, y;
        #ifdef __CUDA_ARCH__ // macro true on device and false on host.
            thrust::default_random_engine rng(clock());
            thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
            rng.discard(id_or_seed);
            x = dist(rng);
            y = dist(rng);
        #else
            unsigned int rseed_ = id_or_seed;
            x = (float)rand_r(&rseed_)/RAND_MAX;
            y = (float)rand_r(&rseed_)/RAND_MAX;
        #endif
        return (x * x + y * y) <= 1.0f;
    };

    std::cout << "calculate pi \n";

    // GPU parallelization ####################################################
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

    // CPP no parallelization #################################################
    gettimeofday(&t1, 0);
    pi_count = thrust::transform_reduce(thrust::cpp::par,
        thrust::counting_iterator<UINT64>(rseed),
        thrust::counting_iterator<UINT64>(rseed + N),
        genpit,
        0.0f, thrust::plus<float>());
    gettimeofday(&t2, 0);
    //std::cout << "CPP pi count: " << pi_count << std::endl;
    pi = pi_count * 4.f / (float) N;
    print_msg("CPP", pi);

    // OMP parallelization ####################################################
    gettimeofday(&t1, 0);
    pi_count = thrust::transform_reduce(thrust::omp::par,
        thrust::counting_iterator<UINT64>(rseed),
        thrust::counting_iterator<UINT64>(rseed  + N),
        genpit,
        0.0f, thrust::plus<float>());
    gettimeofday(&t2, 0);
    pi = pi_count * 4.f / (float) N;
    print_msg("THRUST OMP", pi);

    // Manual OMP parallelization #############################################
    // The thrust::omp::par profile seems slow.
    pi_count = 0;
    gettimeofday(&t1, 0);
    #pragma omp parallel for reduction(+:pi_count)
    for (UINT64 i = 0; i < N; i++) {
        pi_count += genpit(rseed + i);
    }
    gettimeofday(&t2, 0);
    pi = pi_count * 4.f / (float) N;
    print_msg("MANUAL OMP", pi);

    // ########################################################################
    //unsigned int total_threads = std::thread::hardware_concurrency();
    //std::cout << "\nTotal threads " << total_threads << std::endl;

    //std::stringstream msg;
    //msg.str(std::string());
    //msg << "OMP using " << nthreads << " threads.";
    //print_msg(msg.str(), pi);
    int nthreads = omp_get_max_threads();  // use OMP_NUM_THREADS to vary.
    std::cout << "\nOMP using " << nthreads << " threads." << std::endl;

    return 0;
}
