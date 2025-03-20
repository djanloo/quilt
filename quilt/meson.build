py = import('python').find_installation(pure: false)

python_sources = [
  'builder.py',
  'eeg.py',
  'utils.py'
]

install_data('__qinit__.py', 
            rename: '__init__.py',
            install_dir : py.get_install_dir() / 'quilt')

py.install_sources(
  python_sources,
  subdir: 'quilt'
)

subdir('core')
subdir('interface')