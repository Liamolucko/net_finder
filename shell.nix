{
  pkgs ? import <nixpkgs> { },
}:
let
  # Use a patched version of openocd with DirtyJTAG support.
  openocd = pkgs.openocd.overrideAttrs (old: {
    # ./bootstrap isn't included in release tarballs
    src = pkgs.fetchgit {
      url = "https://git.code.sf.net/p/openocd/code";
      rev = "v0.12.0";
      fetchSubmodules = false;
      hash = "sha256-z0bNDPDLBEFxNtKsDWu6n8YRn1NzULxCz4bnSn8Iiyc=";
    };
    patches = [
      (pkgs.fetchpatch {
        url = "https://review.openocd.org/changes/openocd~7344/revisions/23/patch?download";
        decode = "base64 -d";
        hash = "sha256-6CxMJ6l933CV7V88GRCvE/UtYV30BaQ08byf14MGurE=";
      })
    ];
    preConfigure = "./bootstrap nosubmodule";
    nativeBuildInputs = old.nativeBuildInputs ++ [
      pkgs.autoconf
      pkgs.automake
      pkgs.libtool
      pkgs.which
    ];
  });
in
pkgs.mkShell {
  venvDir = ".venv";
  packages = [
    pkgs.python311.pkgs.venvShellHook

    openocd
    pkgs.yosys
    pkgs.verilator

    # Needed by Verilator simulations
    pkgs.json_c
    pkgs.libevent
    pkgs.zlib
  ];

  postVenvCreation = ''
    ${pkgs.uv}/bin/uv pip install -r requirements.txt
  '';
}
