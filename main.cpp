#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <OpenCL/opencl.hpp>
#include <iostream>
#include <cassert>
#include <numeric>
#include<fstream>
#include<string>
#include<random>
#include<algorithm>

cl::Program program;
cl::Context context;
cl::Device device;
class Random
{
public:
    static int get_random_number()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distr(1, 3);
        return distr(gen);
    }
    static void generate_random_matrix(std::vector<float>& A)
    {
        for(int i=0;i<A.size();i++)
        {
            A[i]=get_random_number();
        }
    }
};


void calculate_c(std::vector<float>& m,int col,int row)
{   int counter=0;
    for(int i=0;i<col;i++)
    {
        for(int j=0;j<row;j++)
        {
            m[counter]=1.0/(pow(i+1,2)+j+1);
            counter++;
        }
    }
}

void calculate_b(std::vector<float>& m)
{
    for(int i=0;i<m.size();i++)
    {
        m[i]=5*pow(i+1,3);
    }
}

void print_matrix(std::vector<float> v,int row,int col)
{
    int counter=0;
    for(int i=0;i<col;i++)
    {
        for(int j=0;j<row;j++)
        {
            std::cout<<"["<<v[counter]<<"] ";
            counter++;
        }
        std::cout<<std::endl;
    }
}
void print_vector(std::vector<float> v,int n)
{

    for(int i=0;i<n;i++)
    {
        std::cout<<"["<<v[i]<<"] ";
    }
    std::cout<<std::endl;
}


void multiply_matrix(cl::Kernel& multiply_kernel, cl::CommandQueue& queue,cl::Context context,cl::Buffer& ABuf, cl::Buffer& BBuf,cl::Buffer& CBuf,float* c,int row1,int col1,int col2)
{
    multiply_kernel.setArg(0,ABuf);
    multiply_kernel.setArg(1,BBuf);
    multiply_kernel.setArg(2,CBuf);
    multiply_kernel.setArg(3,row1);
    multiply_kernel.setArg(4,col2);
    multiply_kernel.setArg(5,col1);
    queue.enqueueNDRangeKernel(multiply_kernel,cl::NullRange,cl::NDRange(row1,col2));
    queue.enqueueReadBuffer(CBuf, CL_TRUE, 0, sizeof(int) * row1*col2, c);

}

cl::Device getDefaultDevice(){
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()){
        std::cerr << "No platforms found!" << std::endl;
        exit(1);
    }
    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()){
        std::cerr << "No devices found!" << std::endl;
        exit(1);
    }
    return devices.front();
}


void multiplyByConst(cl::Kernel& kernel,cl::CommandQueue& queue,cl::Context& context,cl::Buffer& input,const float val,cl::Buffer& res,const int input_size,float* c)
{
    kernel.setArg(0,input);
    kernel.setArg(1,res);
    kernel.setArg(2,val);
    queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(input_size));
    queue.enqueueReadBuffer(res, CL_TRUE, 0, sizeof(int) * input_size, c);
}

void sum_vector(cl::Kernel& kernel,cl::CommandQueue& queue,cl::Context& context,cl::Buffer& input1,cl::Buffer& input2,const int input_size,cl::Buffer& output,float* c)
{
    kernel.setArg(0,input1);
    kernel.setArg(1,input2);
    kernel.setArg(2,output);
    queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(input_size));
    queue.enqueueReadBuffer(output, CL_TRUE, 0, sizeof(int) * input_size, c);
}

void subtract_vector(cl::Kernel& kernel,cl::CommandQueue& queue,cl::Context& context,cl::Buffer& input1,cl::Buffer& input2,const int input_size,cl::Buffer& output,float* c)
{
    kernel.setArg(0,input1);
    kernel.setArg(1,input2);
    kernel.setArg(2,output);
    queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(input_size));
    queue.enqueueReadBuffer(output, CL_TRUE, 0, sizeof(int) * input_size, c);
}

void addConst(cl::Kernel& kernel,cl::CommandQueue& queue,cl::Context& context,cl::Buffer& input,const float val,cl::Buffer& res,const int input_size,float* c)
{
    kernel.setArg(0,input);
    kernel.setArg(1,res);
    kernel.setArg(2,val);
    queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(input_size));
    queue.enqueueReadBuffer(res, CL_TRUE, 0, sizeof(int) * input_size, c);
}

int main() {

    device = getDefaultDevice();
    std::cout<<"Type matrix size: ";
    int n;
    std::cin>>n;
    clock_t begin=clock();
    auto vendor = device.getInfo<CL_DEVICE_VENDOR>();
    auto version = device.getInfo<CL_DEVICE_VERSION>();

    std::cout << "Device Vendor: " << vendor << std::endl;
    std::cout << "Device Version: " << version << std::endl;

    context=cl::Context(device);
    cl::Program::Sources sources;

    std::fstream f("../multiply_matrix.cl");
    std::string multiply_code((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());
    sources.push_back({multiply_code.c_str(), multiply_code.length()});
    cl_int exitcode = 0;
    program=cl::Program(context, sources, &exitcode);
    program.build();
    assert(exitcode == CL_SUCCESS);
    cl::Kernel kernel(program, "multiplyMatrices", &exitcode);
    cl::Kernel multiply(program,"multiplyByColum",&exitcode);
    cl::Kernel multiplyByValue(program,"multiplyByConst",&exitcode);
    cl::Kernel sum(program,"sumArrays",&exitcode);
    cl::Kernel subtract(program,"subtractArrays",&exitcode);
    cl::Kernel add_const(program,"addConst",&exitcode);
    assert(exitcode == CL_SUCCESS);

    auto workGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
    std::cout << "Kernel Work Group Size: " << workGroupSize << std::endl;

    std::vector<float> A(n*n);
    std::vector<float> b(n);
    Random::generate_random_matrix(A);
    calculate_b(b);
    std::vector<float> y1(n);
    cl::Buffer ABuf(context,
                    CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                    sizeof(int) * n*n,
                    A.data());

    cl::Buffer bBuf(context,
                    CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                    sizeof(int) * n,b.data());

    cl::Buffer Y1Buf(context,
                     CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
                     sizeof(int) * n);

    cl::CommandQueue queue(context, device,CL_QUEUE_PROFILING_ENABLE);
    cl::Event event;

    multiply_matrix(multiply,queue,context,ABuf,bBuf,Y1Buf,y1.data(),n,n,1);

    std::cout<<"y1= ";
    print_vector(y1,n);
    std::cout<<"Program run: "<<static_cast<double>((clock()-begin))/CLOCKS_PER_SEC<<" seconds";

    return 0;
}