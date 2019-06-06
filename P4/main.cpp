#include <iostream>
#include "timer.h"
#include <string>
#include <stdio.h>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>

// Functions from tonemapping.cu
void preProcess(float **d_luminance, unsigned int **d_cdf,
                size_t *numRows, size_t *numCols, unsigned int *numBins,
                const std::string& filename);

void postProcess(const std::string& output_file, size_t numRows, size_t numCols,
                 float min_logLum, float max_logLum);

void cleanupGlobalMemory(void);

// Function from funcHDR.cu
void calculate_cdf(const float* const d_luminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins);

int main(int argc, char **argv) {
  float *d_luminance;
  unsigned int *d_cdf;

  size_t numRows, numCols;
  unsigned int numBins;

  std::string input_file;
  std::string output_file;
  std::string reference_file;
  double perPixelError = 0.0;
  double globalError   = 0.0;
  bool useEpsCheck = false;

  switch (argc)
  {
	case 2:
	  input_file = std::string(argv[1]);
	  output_file = "output.png";
	  break;
	case 3:
	  input_file  = std::string(argv[1]);
      output_file = std::string(argv[2]);
	  break;
	default:
      std::cerr << "Usage: ./tone input_file [output_filename] [reference_filename]" << std::endl;
      exit(1);
  }

  //load the image and give us our input and output pointers
  preProcess(&d_luminance, &d_cdf,
             &numRows, &numCols, &numBins, input_file);

  GpuTimer timer;
  float min_logLum, max_logLum;
  min_logLum = 0.f;
  max_logLum = 1.f;
  timer.Start();

  //call the students' code
  calculate_cdf(d_luminance, d_cdf, min_logLum, max_logLum,
                               numRows, numCols, numBins);
  timer.Stop();
  cudaDeviceSynchronize(); 
  int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  float *h_luminance = (float *) malloc(sizeof(float)*numRows*numCols);
  unsigned int *h_cdf = (unsigned int *) malloc(sizeof(unsigned int)*numBins);

  cudaMemcpy(h_luminance, d_luminance, numRows*numCols*sizeof(float), cudaMemcpyDeviceToHost);

  //check results and output the tone-mapped image
  postProcess(output_file, numRows, numCols, min_logLum, max_logLum);

  cleanupGlobalMemory();

  return 0;
}
