{
  description = "Develop Python on Nix with uv";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { 
            inherit system;
            config.allowUnfree = true;
          };
        in
        {
          default = (pkgs.buildFHSEnv {
            name = "python-fhs-env";

            targetPkgs = pkgs: with pkgs; [
              python3
              uv
              ninja
              cmake
              ccache
              llvmPackages.libcxxClang
              llvmPackages.libcxx
              cudatoolkit
              cudaPackages.cudnn
              cudaPackages.nccl
              libGL
              libGLU
              libXfixes
              libXft
              fontconfig
              # llvmPackages.lld
              # libllvm
              # glibc
              # glibc.dev
              # binutils
              zenity
              gdb
            ];

            runScript = "nu";

            profile = ''
              # Make CUDA + OpenGL visible
              export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
              export LIBRARY_PATH=/run/opengl-driver/lib:$LIBRARY_PATH
              export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib:$TRITON_LIBCUDA_PATH

              # Make OpenMP visible to puffer env builder
              export EXTRA_CFLAGS="-I${pkgs.llvmPackages.openmp.dev}/include"
              export EXTRA_LDFLAGS="-L${pkgs.llvmPackages.openmp}/lib"

              # export LD_PRELOAD=$(clang -print-file-name=libclang_rt.asan-x86_64.so)

              export CC=clang
              export CXX=clang++

              # manylinux compatibility
              export LD_LIBRARY_PATH=${
                with pkgs; lib.makeLibraryPath (
                  pythonManylinuxPackages.manylinux1 ++ [
                    libXxf86vm
                    libXrandr
                    libXinerama
                    libXcursor
                ])}:$LD_LIBRARY_PATH


              unset PYTHONPATH

              if [ -f ./.venv/bin/activate ]; then
                source .venv/bin/activate
              fi
            '';
          }).env;
        }
      );
    };
}
