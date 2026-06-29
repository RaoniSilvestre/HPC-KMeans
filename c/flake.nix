{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };

      nvhpc-sdk = pkgs.stdenv.mkDerivation {
        pname = "nvhpc-sdk";
        version = "26.5";

        src = pkgs.requireFile {
          name = "nvhpc_2026_265_Linux_x86_64_cuda_13.2.tar.gz";
          url = "https://developer.download.nvidia.com/hpc-sdk/26.5/nvhpc_2026_265_Linux_x86_64_cuda_13.2.tar.gz";
          hash = "sha256-iEmzkJ9+dd9MMq9NxYW/BwHYVJXaQyIbCQqa933yA7Y=";
        };

        dontBuild = true;

        nativeBuildInputs = with pkgs; [
          coreutils
          util-linux
        ];

        installPhase = ''
          mkdir -p $out/opt/nvidia/hpc_sdk
          NVHPC_SILENT=true NVHPC_INSTALL_DIR=$out/opt/nvidia/hpc_sdk bash install
        '';
      };

      fhs = pkgs.buildFHSEnv {
        name = "nvidia-hpc-env";
        targetPkgs = pkgs:
          with pkgs; [
            nvhpc-sdk
            zlib
            zstd

            gdb
            bear
            gnumake

            clang
            openmpi
            llvmPackages.openmp
            cudatoolkit
          ];
        profile = ''
          export NVHPC=/opt/nvidia/hpc_sdk
          export NVHPC_ROOT=$NVHPC/Linux_x86_64/26.5
          export NVCOMPILERS=$NVHPC
          export PATH=$NVHPC_ROOT/compilers/bin:$PATH
        '';
        extraOutputsToInstall = ["man" "doc" "lib" "dev"];
      };
    in {
      devShells.default = fhs.env;
    });
}
