/* Code was based on some code form NVIDIA.*/

/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>
// #include <nppi.h>

#include <helper_cuda.h>
#include <helper_string.h>

bool printfNPPinfo(int argc, char *argv[])
{
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}


int main(int argc, char *argv[])
{
  printf("%s Starting...\n\n", argv[0]);

  try
  {
    std::string sFilename;
    char *filePath;

    findCudaDevice(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false)
    {
      exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input"))
    {
      getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
    }
    else
    {
      filePath = sdkFindFilePath("examples/5.2.10.tiff", argv[0]);
    }

    int states;
    if (checkCmdLineFlag(argc, (const char **)argv, "states"))
    {
      char *states_str;
      getCmdLineArgumentString(argc, (const char **)argv, "states", &states_str);
      states = std::stoi(states_str);
    }
    else
    {
      states = 4;
    }
    std::cout << "states: " << states << "\n";


    if (filePath)
    {
      sFilename = filePath;
    }
    else
    {
      sFilename = "examples/5.2.10.tiff";
    }

    // if we specify the filename at the command line, then we only test
    // sFilename[0].
    int file_errors = 0;
    std::ifstream infile(sFilename.data(), std::ifstream::in);

    if (infile.good())
    {
      std::cout << "boxFilterNPP opened: <" << sFilename.data()
                << "> successfully!" << std::endl;
      file_errors = 0;
      infile.close();
    }
    else
    {
      std::cout << "boxFilterNPP unable to open: <" << sFilename.data() << ">"
                << std::endl;
      file_errors++;
      infile.close();
    }

    if (file_errors > 0)
    {
      exit(EXIT_FAILURE);
    }

    std::string sResultFilename = sFilename;

    std::string::size_type dot = sResultFilename.rfind('.');

    if (dot != std::string::npos)
    {
      sResultFilename = sResultFilename.substr(0, dot);
    }

    sResultFilename += ".out.png";

    if (checkCmdLineFlag(argc, (const char **)argv, "output"))
    {
      char *outputFilePath;
      getCmdLineArgumentString(argc, (const char **)argv, "output",
                               &outputFilePath);
      sResultFilename = outputFilePath;
    }

    NppStreamContext ctx = {0};

    // Device ophalen (meestal device 0)
    int dev;
    cudaGetDevice(&dev);
    ctx.nCudaDeviceId = dev;

    // Stream ophalen (vaak 0 voor de default stream)
    ctx.hStream = 0;

    // Device properties ophalen
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    ctx.nMultiProcessorCount    = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock     = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock      = prop.sharedMemPerBlock;





    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image from disk
    npp::loadImage(sFilename, oHostSrc);
    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);



    int masksize_x = 3;
    int masksize_y = 3;
    // create struct with box-filter mask size
    NppiSize oMaskSize = {masksize_x, masksize_y};

    NppiPoint oSrcOffset = {0, 0};


    // create struct with ROI size
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
    // set anchor point inside the mask to (oMaskSize.width / 2,
    // oMaskSize.height / 2) It should round down when odd
    NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};

    NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};


    npp::ImageNPP_8u_C1 *pSrc = &oDeviceSrc ;
    npp::ImageNPP_8u_C1 *pDst = &oDeviceDst ;

    // discretize image
    int no_of_values = states;
    int total_number_of_values = (1 << 8) ;
    cudaDeviceSynchronize();
    NPP_CHECK_NPP(nppiDivC_8u_C1RSfs_Ctx(
        pSrc->data(), pSrc->pitch(), total_number_of_values/(no_of_values-1), pDst->data(), pDst->pitch(),
        oSizeROI, 0, ctx));

    // Do something fun with CAs
    // e.g. Module sum of neighbors
    npp::ImageCPU_32s_C1 hostKernel(masksize_x,masksize_y);
    for(int x = 0 ; x < masksize_x; x++){
        for(int y = 0 ; y < masksize_y; y++){
            hostKernel.pixels(x,y)[0].x = 0;
        }
    }
    hostKernel.pixels(1,0)[0].x = 1;
    hostKernel.pixels(0,1)[0].x = 1;
    hostKernel.pixels(1,1)[0].x = 1;
    hostKernel.pixels(2,1)[0].x = 1;
    hostKernel.pixels(1,2)[0].x = 1;
    npp::ImageNPP_32s_C1 pKernel(hostKernel);

    cudaDeviceSynchronize();
    std::swap(pSrc, pDst);
    // sum neighbors
    NPP_CHECK_NPP(nppiFilter_8u_C1R_Ctx(
        pSrc->data(), pSrc->pitch(), pDst->data(), pDst->pitch(),
        oSizeROI, pKernel.data(), oMaskSize, oAnchor,
        1, ctx));
    // // get divisor times number of states and set in dst
    std::swap(pSrc, pDst);
    cudaDeviceSynchronize();
    NPP_CHECK_NPP(nppiDivC_8u_C1RSfs_Ctx(
        pSrc->data(), pSrc->pitch(), states, pDst->data(), pDst->pitch(),
        oSizeROI, 0, ctx));
    cudaDeviceSynchronize();
    NPP_CHECK_NPP(nppiMulC_8u_C1RSfs_Ctx(
        pDst->data(), pDst->pitch(), states, pDst->data(), pDst->pitch(),
        oSizeROI, 0, ctx));
    // get the modulus by subtracting divisor times number of states
    cudaDeviceSynchronize();
    NPP_CHECK_NPP(nppiSub_8u_C1RSfs_Ctx(
        pSrc->data(), pSrc->pitch(), pDst->data(), pDst->pitch(),
        pDst->data(), pDst->pitch(), oSizeROI, 0, ctx));


    // convert back to the full range of colours
    cudaDeviceSynchronize();
    std::swap(pSrc, pDst);
    NPP_CHECK_NPP(nppiMulC_8u_C1RSfs_Ctx(
        pSrc->data(), pSrc->pitch(), total_number_of_values/(no_of_values-1), pDst->data(), pDst->pitch(),
        oSizeROI, 0, ctx));
    cudaDeviceSynchronize();
    // std::swap(pSrc, pDst);


    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHostDst(pDst->size());
    // and copy the device result data into it
    pDst->copyTo(oHostDst.data(), oHostDst.pitch());

    saveImage(sResultFilename, oHostDst);
    std::cout << "Saved image: " << sResultFilename << std::endl;

    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());

    exit(EXIT_SUCCESS);
  }
  catch (npp::Exception &rException)
  {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  }
  catch (...)
  {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}
