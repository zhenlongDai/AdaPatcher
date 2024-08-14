set pagination off
set logging file trace.log
set logging enabled on

break main.cpp:7
commands
  silent
  printf "At main.cpp:7\n"
  info locals
  continue
end

break main.cpp:8
commands
  silent
  printf "At main.cpp:8\n"
  info locals
  continue
end

break main.cpp:9
commands
  silent
  printf "At main.cpp:9\n"
  info locals
  continue
end

break main.cpp:12
commands
  silent
  printf "At main.cpp:12\n"
  info locals
  continue
end

run
quit
