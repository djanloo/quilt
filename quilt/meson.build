fs=import('fs')

py = import('python').find_installation(pure: false)

python_sources = [
  '__init__.py',
  'builder.py'
]
py.install_sources(
  python_sources,
  subdir: 'quilt'
)

subdir('core')
subdir('interface')