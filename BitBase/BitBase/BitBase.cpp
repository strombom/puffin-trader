#include "pch.h"

#include "Logger.h"
#include "Bitmex.h"
#include "Binance.h"
#include "Server.h"
#include "Database.h"
#include "DownloadManager.h"
#include "BitBotConstants.h"

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
    auto cmd = std::string{};
    std::cin >> cmd;
    return cmd;
}

int main()
{
    logger.info("BitBase started");

    SetConsoleCtrlHandler(CtrlHandler, TRUE);

    //auto download_manager = DownloadManager::create();
    auto database = Database::create(BitBase::Database::root_path);
    //auto bitmex = Bitmex{ database, download_manager };
    auto binance = Binance{ database };
    auto server = Server{ database };

    auto keyboard_input = std::async(get_keyboard_input);
    while (running) {
        if (keyboard_input.wait_for(std::chrono::milliseconds{ 500 }) == std::future_status::ready) {
            std::string cmd = keyboard_input.get();
            if (cmd == "q" || cmd == "quit") {
                running = false;
                break;
            } else {
                logger.info("Unknown command (%s)", cmd.c_str());
            }
            keyboard_input = std::async(get_keyboard_input);
        }
    }
    logger.info("Shutting down");

    //bitmex.shutdown();
    binance.shutdown();
    //download_manager->shutdown();
}
