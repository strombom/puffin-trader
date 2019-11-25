
#include "Logger.h"
#include "Bitmex.h"
#include "Database.h"
#include "DownloadManager.h"

#include <future>
#include <iostream>
#include <windows.h> 


bool running = true;

BOOL WINAPI CtrlHandler(DWORD fdwCtrlType)
{
    switch (fdwCtrlType)
    {
        // Handle the CTRL-C signal. 
    case CTRL_C_EVENT:
        logger.info("ctrl-c pressed");
        running = false;
        return TRUE;

        // CTRL-CLOSE: confirm that the user wants to exit. 
    case CTRL_CLOSE_EVENT:
        logger.info("ctrl-close pressed");
        running = false;
        return TRUE;

        // Pass other signals to the next handler. 
    case CTRL_BREAK_EVENT:
        logger.info("ctrl-break pressed");
        running = false;
        return TRUE;

    case CTRL_LOGOFF_EVENT:
        logger.info("ctrl-logoff pressed");
        running = false;
        return TRUE;

    case CTRL_SHUTDOWN_EVENT:
        logger.info("ctrl-shutdown pressed");
        running = false;
        return TRUE;

    default:
        return FALSE;
    }
}

static std::string get_keyboard_input()
{
    std::string cmd;
    std::cin >> cmd;
    return cmd;
}

int main()
{
    SetConsoleCtrlHandler(CtrlHandler, TRUE);

    sptrDownloadManager download_manager = DownloadManager::create();
    sptrDatabase database = Database::create("C:\\development\\github\\puffin-trader\\database\\data");

    Bitmex bitmex(database, download_manager);
    
    std::future<std::string> future = std::async(get_keyboard_input);
    while (running) {
        if (future.wait_for(std::chrono::milliseconds{ 500 }) == std::future_status::ready) {
            std::string cmd = future.get();
            if (cmd == "q" || cmd == "quit") {
                running = false;
                break;
            } else {
                logger.info("Unknown command (%s)", cmd.c_str());
            }
            future = std::async(get_keyboard_input);
        }
    }
    logger.info("Shutting down");

    download_manager->shutdown();
    bitmex.shutdown();
}
