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

            targetPkgs = pkgs: [
              pkgs.python3
              pkgs.uv
              pkgs.ninja
              pkgs.cudatoolkit
              pkgs.binutils
              pkgs.libllvm
              pkgs.zenity
            ];

            multiPkgs = pkgs: [
              pkgs.stdenv.cc
              pkgs.llvmPackages.clang
              pkgs.llvmPackages.lld
              pkgs.glibc
              pkgs.glibc.dev
            ];

            runScript = "nu";

            profile = ''
              # Make CUDA + OpenGL visible
              export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
              export LIBRARY_PATH=/run/opengl-driver/lib:$LIBRARY_PATH
              export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib:$TRITON_LIBCUDA_PATH

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
          # default = pkgs.mkShell {
          #   packages = [
          #     pkgs.python3
          #     pkgs.uv
          #     pkgs.ninja
          #     pkgs.cudatoolkit
          #   ];
          #
          #   env = lib.optionalAttrs pkgs.stdenv.isLinux {
          #     # Python libraries often load native shared objects using dlopen(3).
          #     # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
          #     LD_LIBRARY_PATH = lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
          #   };
          #
          #   shellHook = ''
          #     # Required for Pytorch to find cuda libraries
          #     export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
          #     export LIBRARY_PATH=/run/opengl-driver/lib:$LIBRARY_PATH
          #     export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib:$TRITON_LIBCUDA_PATH
          #
          #     unset PYTHONPATH
          #     if [ -f ./.venv/bin/activate ]; then
          #       source .venv/bin/activate
          #     fi
          #   '';
          # };
        }
      );
    };
}
