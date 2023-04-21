use GPUAPI;
use CTypes;

extern proc kernel(dA: c_void_ptr, n: int);

var D = {0..31};
var A: [D] int;
var V: [D] int; // for verification

// initialization proc
proc init(arr: [?dom] int) {
    for i in dom {
        arr[i] = i;
    }
    V = arr + 1;
}

// MID-LOW
// dA is a linearized 1D GPU array
init(A);

var dA: c_void_ptr;
var size: c_size_t = A.size:c_size_t * c_sizeof(A.eltType);
Malloc(dA, size);
Memcpy(dA, c_ptrTo(A), size, 0);
kernel(dA, D.dim(0).size);
DeviceSynchronize();
Memcpy(c_ptrTo(A), dA, size, 1);

// Verify
if (A.equals(V)) {
    writeln("MID-LOW Verified");
} else {
    writeln("MID-LOW Not Verified");
}

// MID
init(A);

var dA2 = new GPUArray(A);
dA2.toDevice();

kernel(dA2.dPtr(), D.dim(0).size);
DeviceSynchronize();
dA2.fromDevice();

// Verify
if (A.equals(V)) {
    writeln("MID Verified");
} else {
    writeln("MID Not Verified");
}

// MID (unified memory)
var uA = new GPUUnifiedArray(int, D);
init(uA.a);

kernel(uA.dPtr(), D.dim(0).size);
DeviceSynchronize();

// Verify
if (uA.a.equals(V)) {
    writeln("MID (unified memory) Verified");
} else {
    writeln("MID (unified memory) Not Verified");
}

// MID (unified memory - Chapel array)
use GPUUnifiedDist;
var uD = D dmapped GPUUnified({0..31});
var uA2: [uD] int;
init(uA2);

kernel(c_ptrTo(uA2.localSlice(uD.localSubdomain(here))), uD.dim(0).size);
DeviceSynchronize();

// Verify
if (uA2.equals(V)) {
    writeln("MID (unified memory - Chapel array) Verified");
} else {
    writeln("MID (unified memory - Chapel array) Not Verified");
}
