.PHONY: generate exe

LIBDIR := /usr/lib/x86_64-linux-gnu

generate:
	meson setup build -Dpython.install_env=venv -Dlibdir=$(LIBDIR)
	meson compile -C build
	meson install -C build

clean:
	rm -rf build

exe:
	meson setup build -Dpython.install_env=venv -Dlibdir=$(LIBDIR)
	meson compile quilt.exe -C build 
	cp build/quilt/core/quilt.exe ./