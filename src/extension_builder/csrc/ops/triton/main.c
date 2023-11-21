
#include "cuda.h"
#include <stdbool.h>
#include <Python.h>
#include <dlfcn.h>

static inline void gpuAssert(CUresult code, const char *file, int line)
{
   if (code != CUDA_SUCCESS)
   {
      const char* prefix = "Triton Error [CUDA]: ";
      const char* str;
      cuGetErrorString(code, &str);
      char err[1024] = {0};
      strcat(err, prefix);
      strcat(err, str);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }
}

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

typedef CUresult (*cuLaunchKernelEx_t)(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra);

static cuLaunchKernelEx_t getLaunchKernelExHandle() {
  // Open the shared library
  void* handle = dlopen("libcuda.so", RTLD_LAZY);
  if (!handle) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so");
    return NULL;
  }
  // Clear any existing error
  dlerror();
  cuLaunchKernelEx_t cuLaunchKernelExHandle = (cuLaunchKernelEx_t)dlsym(handle, "cuLaunchKernelEx");
  // Check for errors
  const char *dlsym_error = dlerror();
  if (dlsym_error) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to retrieve cuLaunchKernelEx from libcuda.so");
    return NULL;
  }
  return cuLaunchKernelExHandle;
}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, CUstream stream, CUfunction function, CUdeviceptr arg0, CUdeviceptr arg1, CUdeviceptr arg2, int32_t arg3, int32_t arg4, int32_t arg5, int32_t arg6, int32_t arg7, int32_t arg8, int32_t arg9, int32_t arg10, int32_t arg11) {
  void *params[] = { &arg0, &arg1, &arg2, &arg3, &arg4, &arg5, &arg6, &arg7, &arg8, &arg9, &arg10, &arg11 };
  if (gridX*gridY*gridZ > 0) {
    if (num_ctas == 1) {
      CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0));
    } else {
      CUlaunchAttribute launchAttr[2];
      launchAttr[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launchAttr[0].value.clusterDim.x = clusterDimX;
      launchAttr[0].value.clusterDim.y = clusterDimY;
      launchAttr[0].value.clusterDim.z = clusterDimZ;
      launchAttr[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launchAttr[1].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
      CUlaunchConfig config;
      config.gridDimX = gridX * clusterDimX;
      config.gridDimY = gridY * clusterDimY;
      config.gridDimZ = gridZ * clusterDimZ;
      config.blockDimX = 32 * num_warps;
      config.blockDimY = 1;
      config.blockDimZ = 1;
      config.sharedMemBytes = shared_memory;
      config.hStream = stream;
      config.attrs = launchAttr;
      config.numAttrs = 2;
      static cuLaunchKernelEx_t cuLaunchKernelExHandle = NULL;
      if (cuLaunchKernelExHandle == NULL) {
        cuLaunchKernelExHandle = getLaunchKernelExHandle();
      }
      CUDA_CHECK(cuLaunchKernelExHandle(&config, function, params, 0));
    }
  }
}

typedef struct _DevicePtrInfo {
    CUdeviceptr dev_ptr;
    bool valid;
} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    uint64_t dev_ptr;
    int status = cuPointerGetAttribute(&dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == CUDA_ERROR_INVALID_VALUE) {
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }
    ptr_info.dev_ptr = dev_ptr;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}

static PyObject* launch(PyObject* self, PyObject* args) {
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int num_warps;
  int num_ctas;
  int clusterDimX;
  int clusterDimY;
  int clusterDimZ;
  int shared_memory;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *compiled_kernel = NULL;
  PyObject* _arg0;  PyObject* _arg1;  PyObject* _arg2;  int32_t _arg3;  int32_t _arg4;  int32_t _arg5;  int32_t _arg6;  int32_t _arg7;  int32_t _arg8;  int32_t _arg9;  int32_t _arg10;  int32_t _arg11; 
  if(!PyArg_ParseTuple(args, "iiiiiiiiiKKOOOOOOiiiiiiiii", &gridX, &gridY, &gridZ, &num_warps, &num_ctas, &clusterDimX, &clusterDimY, &clusterDimZ, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel, &_arg0, &_arg1, &_arg2, &_arg3, &_arg4, &_arg5, &_arg6, &_arg7, &_arg8, &_arg9, &_arg10, &_arg11)) {
    return NULL;
  }

  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {
    return NULL;
  }


  // raise exception asap
  DevicePtrInfo ptr_info0 = getPointer(_arg0, 0); if (!ptr_info0.valid) return NULL;; DevicePtrInfo ptr_info1 = getPointer(_arg1, 1); if (!ptr_info1.valid) return NULL;; DevicePtrInfo ptr_info2 = getPointer(_arg2, 2); if (!ptr_info2.valid) return NULL;; ; ; ; ; ; ; ; ; ;
  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (CUstream)_stream, (CUfunction)_function, ptr_info0.dev_ptr, ptr_info1.dev_ptr, ptr_info2.dev_ptr, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8, _arg9, _arg10, _arg11);
  Py_END_ALLOW_THREADS;
  if (PyErr_Occurred()) {
    return NULL;
  }

  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {
    return NULL;
  }

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__triton_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit___triton_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
