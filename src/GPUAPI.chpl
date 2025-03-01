/*
 * Copyright (c) 2019, Rice University
 * Copyright (c) 2019, Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

module GPUAPI {
    use CTypes;

    config param debugGPUAPI = false;

    extern proc GetDeviceCount(ref count: int(32));
    extern proc GetDevice(ref device: int(32));
    extern proc SetDevice(device: int(32));

    extern proc ProfilerStart();
    extern proc ProfilerStop();

    extern proc DeviceSynchronize();

    // cudaMalloc
    extern proc Malloc(ref devPtr: c_ptr(void), size: c_size_t);
    extern proc MallocPtr(ref devPtr: c_ptr(c_ptr(void)), size: c_size_t);
    extern proc MallocPtrPtr(ref devPtr: c_ptr(c_ptr(c_ptr(void))), size: c_size_t);
    inline proc Malloc(ref devPtr: c_ptr(c_ptr(void)), size: c_size_t) { MallocPtr(devPtr, size); };
    inline proc Malloc(ref devPtr: c_ptr(c_ptr(c_ptr(void))), size: c_size_t) { MallocPtrPtr(devPtr, size); };

    // cudaMallocPitch
    extern proc MallocPitch(ref devPtr: c_ptr(void), ref pitch: c_size_t, width: c_size_t, height: c_size_t);

    // cudaMallocManaged
    extern proc MallocManaged(ref umemPtr: c_ptr(?eltType), size: c_size_t);
    // cudaMemPrefetchAsync
    extern proc PrefetchToDevice(umemPtr: c_ptr(void), start: c_size_t, end: c_size_t, device: int(32));

    extern proc Memcpy(dst: c_ptr(void), src: c_ptr(void), count: c_size_t, kind: int);
    extern proc Memcpy2D(dst: c_ptr(void), dpitch: c_size_t, src: c_ptr(void), spitch: c_size_t, width: c_size_t, height: c_size_t, kind: int);
    extern proc Free(devPtr: c_ptr(void));

    @chpldoc.nodoc
    inline operator c_ptr.+(a: c_ptr(void), b: uint(64)) { return __primitive("+", a, b); }

    class GPUArray {
      var devPtr: c_ptr(void);
      var hosPtr: c_ptr(void);
      var size: c_size_t;
      var sizeInBytes: c_size_t;
      var pitched: bool;
      var hpitch: c_size_t;
      var dpitch: c_size_t;
      var height: c_size_t;

      proc init(ref arr, pitched=false) {
        // Low-level info
        this.devPtr = nil;
        this.hosPtr = c_ptrTo(arr);
        // size info
        size = arr.size: c_size_t;
        sizeInBytes = (((arr.size: c_size_t) * c_sizeof(arr.eltType)) : c_size_t);
        this.pitched = pitched;
        init this;
        if (!pitched) {
            // allocation
            Malloc(devPtr, sizeInBytes);
            if (arr.rank == 2) {
               this.hpitch = arr.domain.dim(1).size:c_size_t * c_sizeof(arr.eltType);
               this.dpitch = this.hpitch;
            }
        } else if (arr.rank == 2) {
            this.hpitch = arr.domain.dim(1).size:c_size_t * c_sizeof(arr.eltType);
            // allocation
            MallocPitch(devPtr, this.dpitch, this.hpitch, arr.domain.dim(0).size:c_size_t);
            this.height = arr.domain.dim(0).size:c_size_t;
        } else {
            writeln("GPU Array allocation error: pitched=true only works with 2D array, but the given rank is ", arr.rank);
            exit();
        }
        if (debugGPUAPI) { writeln("malloc'ed: ", devPtr, " sizeInBytes: ", sizeInBytes); }
      }

      proc deinit() {
          Free(this.dPtr());
      }

      inline proc toDevice() {
          if (this.pitched == false) {
              Memcpy(this.dPtr(), this.hPtr(), this.sizeInBytes, 0);
          } else {
              Memcpy2D(this.dPtr(), this.dpitch, this.hPtr(), this.hpitch, this.hpitch, this.height, 0);
          }
          if (debugGPUAPI) { writeln("h2d : ", this.hPtr(), " -> ", this.dPtr(), " transBytes: ", this.sizeInBytes); }
      }

      inline proc fromDevice() {
          if (this.pitched == false) {
              Memcpy(this.hPtr(), this.dPtr(), this.sizeInBytes, 1);
          } else {
              Memcpy2D(this.hPtr(), this.hpitch, this.dPtr(), this.dpitch, this.hpitch, this.height, 1);
          }
          if (debugGPUAPI) { writeln("d2h : ", this.dPtr(), " -> ", this.hPtr(), " transBytes: ", this.sizeInBytes); }
      }

      inline proc free() {
        Free(this.dPtr());
        if (debugGPUAPI) { writeln("free : ", this.dPtr()); }
      }

      inline proc dPtr(): c_ptr(void) {
        return devPtr;
      }

      inline proc hPtr(): c_ptr(void) {
        return hosPtr;
      }
    }

    inline proc toDevice(args: GPUArray ...?n) {
      for ga in args {
        ga.toDevice();
      }
    }

    inline proc fromDevice(args: GPUArray ...?n) {
      for ga in args {
        ga.fromDevice();
      }
    }

    inline proc free(args: GPUArray ...?n) {
      for ga in args {
        ga.free();
      }
    }

    class GPUJaggedArray {
      var devPtr: c_ptr(c_ptr(void));
      var nRows: int;
      var hosPtrs: [0..#nRows] c_ptr(void);
      var devPtrs: [0..#nRows] c_ptr(void);
      var elemSizes: [0..#nRows] c_size_t;
      var size: c_size_t;
      var sizeInBytes: c_size_t;

      // args: iterable
      // class C {
      //   var n: int;
      //   proc init(_n: int) { n = _n;  }
      //   var x: [0..#n] int;
      // }
      // var Cs = [new C(256), new C(512)];
      // var dCs = new GPUJaggedArray(Cs.x);
      proc init(args) {
        this.nRows = 0;
        for i in args {
            this.nRows = this.nRows + 1;
        }
        init this;
        var idx = 0;
        for i in args {
          const elemSize = i.size:c_size_t * c_sizeof(i.eltType);
          Malloc(this.devPtrs[idx], elemSize);
          this.hosPtrs[idx] = c_ptrTo(i);
          this.elemSizes[idx] = elemSize;
          idx = idx + 1;
        }
        Malloc(this.devPtr, nRows:c_size_t * c_sizeof(c_ptr(c_ptr(void))));
      }

      // var dCs = new GPUJaggedArray(Cs[0].x, Cs[1].x, ...);
      proc init(args ...?n) where n>=2 {
        this.nRows = n;
        init this;
        var idx: int = 0;
        for arg in args {
          const elemSize = arg.size: c_size_t * c_sizeof(arg.eltType);
          Malloc(this.devPtrs[idx], elemSize);
          this.hosPtrs[idx] = c_ptrTo(arg);
          this.elemSizes[idx] = elemSize;
          idx = idx + 1;
        }
        Malloc(this.devPtr, nRows:c_size_t * c_sizeof(c_ptr(c_ptr(void))));
      }

      proc deinit() {
        for i in 0..#nRows {
          Free(this.devPtrs[i]);
        }
        Free(this.devPtr);
      }

      inline proc toDevice() {
        for i in 0..#nRows {
          Memcpy(this.devPtrs[i], this.hosPtrs[i], elemSizes[i], 0);
        }
        Memcpy(this.devPtr, c_ptrTo(this.devPtrs), nRows:c_size_t * c_sizeof(c_ptr(c_ptr(void))), 0);
      }

      inline proc fromDevice() {
        for i in 0..#nRows {
          Memcpy(this.hosPtrs[i], this.devPtrs[i], elemSizes[i], 1);
        }
      }

      inline proc dPtr(): c_ptr(c_ptr(void)) {
        return devPtr;
      }
    }

    class GPUUnifiedArray {
      type etype;
      var umemPtr: c_ptr(etype) = nil;
      var dom: domain; // TODO handle non-zero-based domains
      var a = makeArrayFromPtr(umemPtr, dom);

      proc init(type etype, dom: domain(?)) {
        this.etype = etype;
        this.dom = dom;
        // allocation
        MallocManaged(umemPtr, dom.size * c_sizeof(etype));
        if (debugGPUAPI) { writeln("malloc'ed managed: ", umemPtr, " sizeInBytes: ", dom.size*c_sizeof(etype)); }
      }

      proc deinit() {
        if (debugGPUAPI) { writeln("free : ", c_ptrTo(a)); }
        Free(c_ptrTo(a));
      }

      inline proc dPtr(): c_ptr(void) {
        return c_ptrTo(a);
      }

      /*
      * Get a pointer to the element at the given index..
      */
      inline proc dPtr(i: a.rank*a._value.dom.idxType): c_ptr(void) {
        return c_ptrTo(a(i));
      }

      inline proc dPtr(i: a._value.dom.idxType ...a.rank): c_ptr(void) {
        return dPtr(i);
      }
      
      // TODO handle multi-dimensional array segments (non-contiguous?)
      inline proc prefetchToDevice(startIdx: int, endIdx: int, device: int(32)) {
        if (debugGPUAPI) { writeln("PrefetchToDevice for umemPtr ", c_ptrTo(a), " device ", device, ": ", startIdx*c_sizeof(etype), "..", (endIdx+1)*c_sizeof(etype)); }
        PrefetchToDevice(c_ptrTo(a), startIdx*c_sizeof(etype), (endIdx+1)*c_sizeof(etype), device);
      }
    }
}
