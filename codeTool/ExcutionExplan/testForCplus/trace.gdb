set pagination off
set logging file trace.log
set logging enabled on



# Include breakpoints here


break main.cpp:1
commands
  silent
  printf "At main.cpp:1\n"
  info locals
  continue
end

break main.cpp:3
commands
  silent
  printf "At main.cpp:3\n"
  info locals
  continue
end

break main.cpp:4
commands
  silent
  printf "At main.cpp:4\n"
  info locals
  continue
end

break main.cpp:5
commands
  silent
  printf "At main.cpp:5\n"
  info locals
  continue
end

break main.cpp:6
commands
  silent
  printf "At main.cpp:6\n"
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

break main.cpp:10
commands
  silent
  printf "At main.cpp:10\n"
  info locals
  continue
end

break main.cpp:11
commands
  silent
  printf "At main.cpp:11\n"
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

break main.cpp:13
commands
  silent
  printf "At main.cpp:13\n"
  info locals
  continue
end

break main.cpp:14
commands
  silent
  printf "At main.cpp:14\n"
  info locals
  continue
end

break main.cpp:15
commands
  silent
  printf "At main.cpp:15\n"
  info locals
  continue
end

break main.cpp:16
commands
  silent
  printf "At main.cpp:16\n"
  info locals
  continue
end

break main.cpp:17
commands
  silent
  printf "At main.cpp:17\n"
  info locals
  continue
end

run
quit
