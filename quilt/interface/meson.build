inc_dirs = include_directories('../core')

pyxsources = ['base.pyx',
                        'spiking.pyx',
                        'oscill.pyx',
                        'multiscale.pyx']


dep_py = py.dependency()

cy = meson.get_compiler('cython')
# generator() doesn't accept compilers, only found programs - cast it.
cython = find_program(cy.cmd_array()[0])

py.install_sources(
  'cinterface.pxd',
  subdir: 'quilt/interface'
)

pxd_files = files(
  'some_header.pxd',
  'another_header.pxd',
  # aggiungi tutti i tuoi file .pxd qui
)


foreach cython_source : pyxsources
    name = fs.name(cython_source).split('.')[0]
    cpp_cyth_file = custom_target(name,
    output : name +'.cpp',
    input : name + '.pyx',
    depend_files: [name + '.pxd', 'cinterface.pxd'], 
    command : [cython, '--cplus', '-3', '--fast-fail', '@INPUT@', '-o', '@OUTPUT@']
    )

    cython_cpp_lib = py.extension_module(
        name,
        cpp_cyth_file,
        override_options : ['cython_language=cpp'],
        link_with : [libcpp],
        dependencies : dep_py,
        include_directories : inc_dirs,
        subdir: 'quilt/interface',
        install: true
    )
endforeach