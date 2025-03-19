{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  name = "hrap-env";
  buildInputs = with pkgs; [
    # Python with packages
    (python312.withPackages (
      ps: with ps; [
        numpy
        scipy
        matplotlib
        pytest
        ipython
      ]
    ))
  ];
}
