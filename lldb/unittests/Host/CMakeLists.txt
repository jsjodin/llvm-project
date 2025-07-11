set (FILES
  ConnectionFileDescriptorTest.cpp
  FileActionTest.cpp
  FileSystemTest.cpp
  FileTest.cpp
  HostInfoTest.cpp
  HostTest.cpp
  MainLoopTest.cpp
  JSONTransportTest.cpp
  NativeProcessProtocolTest.cpp
  PipeTest.cpp
  ProcessLaunchInfoTest.cpp
  SocketAddressTest.cpp
  SocketTest.cpp
  ThreadLauncherTest.cpp
  XMLTest.cpp
)

if (UNIX)
  list(APPEND FILES
    posix/HostTest.cpp
    posix/SupportTest.cpp
  )
endif()

if (LLDB_ENABLE_TERMIOS)
  list(APPEND FILES
    posix/TerminalTest.cpp
  )
endif()

add_lldb_unittest(HostTests
  ${FILES}
  LINK_LIBS
    lldbHost
    lldbCore
    lldbUtilityHelpers
    lldbHostHelpers
    LLVMTestingSupport
    LLVMTargetParser
  )

add_subdirectory(common)
