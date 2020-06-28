#include <cassert>
#include "util/gl_wrapper.h"
#include "util/OpenGL32Format.h"
#include <QApplication>
#include <QGLWidget>
#include <Eigen/Dense>
#include "cudax/CublasHelper.h"
#include "cudax/CudaHelper.h"
#include "cudax/CublasHelper.h"

using namespace std;
using namespace cudax;

typedef float Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix_MxN;
extern "C" void outer_product(float* input, float* output, int rows, int cols);

int main(int argc, char *argv[]){
    QApplication app(argc, argv); 
    OpenGL32Format fmt;
    QGLWidget widget(fmt);
    widget.makeCurrent();
    CudaHelper::init();
    CublasHelper::init();
       
    {
//    for(int i=0; i<5;i++) 
	//cout << "item: " << i << dv[i]<<endl;
    	//cout<<"average is" << sum / 0.5f <<endl;
    }
    
    CublasHelper::cleanup();
    CudaHelper::cleanup();
    return 0;						
}
