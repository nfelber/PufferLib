#!/bin/bash
set -e

PLATFORM="$(uname -s)"
if [ "$PLATFORM" = "Linux" ]; then
    RAYLIB_NAME="raylib-5.5_linux_amd64"
    STANDALONE_LDFLAGS=(-lGL)
else
    RAYLIB_NAME="raylib-5.5_macos"
    STANDALONE_LDFLAGS=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
fi

RAYLIB_URL="https://github.com/raysan5/raylib/releases/download/5.5"
if [ ! -d "$RAYLIB_NAME" ]; then
    echo "Downloading $RAYLIB_NAME..."
    curl -sL "$RAYLIB_URL/$RAYLIB_NAME.tar.gz" -o "$RAYLIB_NAME.tar.gz"
    tar xf "$RAYLIB_NAME.tar.gz"
    rm "$RAYLIB_NAME.tar.gz"
fi

${CC:-clang} -O2 -DNDEBUG -Wall \
    -I"./$RAYLIB_NAME/include" \
    tools/quad_meshing_3d_poc.c \
    "./$RAYLIB_NAME/lib/libraylib.a" \
    "${STANDALONE_LDFLAGS[@]}" \
    -lm -lpthread \
    -DPLATFORM_DESKTOP \
    -o quad_meshing_3d_poc

echo "Built: ./quad_meshing_3d_poc"
