
// #ifndef TIME_HELPER_H
// #define TIME_HELPER_H

// #define DEBUG


// #include <map>
// #include <cuda_runtime.h>
// #include <string>
// #include <vector>
// #include <stdarg.h>
// #include <sys/time.h>
// #include <string>

// // #include <helper_cuda.h>
// #include "helper_cuda.h"



// double cpuMilliSecond() {
//     struct timeval tp;
//     gettimeofday(&tp,NULL);  
//     return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
// }

// class Timer{
// public:
//     cudaEvent_t event1, event2;
//     double start_time, cpu_time;
//     cudaEvent_t *event_now, *event_before, *event_tmp;
//     bool absolute;
//     float total_elapse_time=0;
//     Timer(bool absolute_=false, bool init_record=true):absolute(absolute_){
//         checkCudaErrors(cudaEventCreate(&event1)); 
//         checkCudaErrors(cudaEventCreate(&event2));
//         event_now = &event2;
//         event_before = &event1;
//         if (init_record){
//             checkCudaErrors(cudaEventRecord(*event_now, 0));
//             swap();
//         }

//     };
//     float elapse(){
//         float elapsed_time=0;
//         checkCudaErrors(cudaEventRecord(*event_now, 0));
//         checkCudaErrors(cudaEventSynchronize(*event_now));
//         checkCudaErrors(cudaEventElapsedTime(&elapsed_time, *event_before, *event_now));
//         if(! this-> absolute)
//             swap();
//         total_elapse_time += elapsed_time;
//         return elapsed_time;
//     };
//     float total(){
//         return total_elapse_time;
//     }

//     void start(cudaStream_t stream=0){
//         checkCudaErrors(cudaEventRecord(event1, stream));
//         start_time = cpuMilliSecond();
//     }
//     double stop(cudaStream_t stream=0){
//         checkCudaErrors(cudaEventRecord(event2, stream));
//         cpu_time = cpuMilliSecond() - start_time;
//         return cpu_time;
//     }
//     double get_cpu_time(){return cpu_time;}
//     float get_time(){
//         float elapsed_time=0;
//         checkCudaErrors(cudaEventSynchronize(event2));
//         checkCudaErrors(cudaEventElapsedTime(&elapsed_time, event1, event2));
//         return elapsed_time;
//     }

//     ~Timer(){
//         if (cudaEventQuery(event1) != cudaSuccess) 
//             checkCudaErrors(cudaEventDestroy(event1));    //destory the event
//         if (cudaEventQuery(event2) != cudaSuccess) 
//             checkCudaErrors(cudaEventDestroy(event2));
//     }
// private:
//     void swap(){
//         event_tmp = event_now;
//         event_now = event_before;
//         event_before = event_tmp;
//     };
// };

// typedef std::map<std::string, Timer> UDT_MAP_STRING_TIMER;

// void log_printf(const char *fmt, ...){
//     #ifdef DEBUG
//     va_list args;       //定义一个va_list类型的变量，用来储存单个参数  
//     va_start(args, fmt); //使args指向可变参数的第一个参数  
//     vprintf(fmt, args);  //必须用vprintf等带V的  
//     va_end(args);       //结束可变参数的获取
//     #endif
// }



// class TimersMap{
// private:
//     std::map<std::string, Timer> timers;
// public:
//     TimersMap(std::vector<std::string> keys){
//         for (auto key : keys){
//             timers[key] = Timer(false, false);
//         }
//     };
//     void start(std::string key, cudaStream_t stream=0){
//         // std::cout << "start: " << key << std::endl;
//         if (timers.find(key) != timers.end()){
//             timers[key].start(stream);
//         }else{
//             std::cout << key << "not found in timers" << std::endl;
//         }
//     };
//     void stop(std::string key, cudaStream_t stream=0){
//         if (timers.find(key) != timers.end()){
//             timers[key].stop(stream);
//         }else{
//             std::cout << key << "not found in timers" << std::endl;
//         }
//     };
//     void finish_and_report(){        
//         for(auto t=timers.begin(); t != timers.end(); t++){
//         // for(auto t:  timers){
//             printf("%s: %f, %lf\n", t->first.c_str(), t->second.get_time(), t->second.get_cpu_time());
//         }
//     };
// };


// class EventTracer{
// private:
//     std::map<std::string, double> events;
// };

// #endif