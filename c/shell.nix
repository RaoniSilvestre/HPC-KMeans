{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  packages = with pkgs; [
    gnumake
    gcc
    bear
    gdb
    openmpi
    llvmPackages.clang
    llvmPackages.openmp
  ];
  shellHook = ''
    export MANPATH="${pkgs.openmpi.man}/share/man:$MANPATH"
  '';
}
