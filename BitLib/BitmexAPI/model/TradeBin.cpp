/**
 * BitMEX API
 * ## REST API for the BitMEX Trading Platform  [View Changelog](/app/apiChangelog)  -  #### Getting Started  Base URI: [https://www.bitmex.com/api/v1](/api/v1)  ##### Fetching Data  All REST endpoints are documented below. You can try out any query right from this interface.  Most table queries accept `count`, `start`, and `reverse` params. Set `reverse=true` to get rows newest-first.  Additional documentation regarding filters, timestamps, and authentication is available in [the main API documentation](/app/restAPI).  _All_ table data is available via the [Websocket](/app/wsAPI). We highly recommend using the socket if you want to have the quickest possible data without being subject to ratelimits.  ##### Return Types  By default, all data is returned as JSON. Send `?_format=csv` to get CSV data or `?_format=xml` to get XML data.  ##### Trade Data Queries  _This is only a small subset of what is available, to get you started._  Fill in the parameters and click the `Try it out!` button to try any of these queries.  - [Pricing Data](#!/Quote/Quote_get)  - [Trade Data](#!/Trade/Trade_get)  - [OrderBook Data](#!/OrderBook/OrderBook_getL2)  - [Settlement Data](#!/Settlement/Settlement_get)  - [Exchange Statistics](#!/Stats/Stats_history)  Every function of the BitMEX.com platform is exposed here and documented. Many more functions are available.  ##### Swagger Specification  [⇩ Download Swagger JSON](swagger.json)  -  ## All API Endpoints  Click to expand a section. 
 *
 * OpenAPI spec version: 1.2.0
 * Contact: support@bitmex.com
 *
 * NOTE: This class is auto generated by the swagger code generator 2.4.11-SNAPSHOT.
 * https://github.com/swagger-api/swagger-codegen.git
 * Do not edit the class manually.
 */



#include "BitmexAPI/model/TradeBin.h"

namespace io {
namespace swagger {
namespace client {
namespace model {

TradeBin::TradeBin()
{
    m_Timestamp = utility::datetime();
    m_Symbol = utility::conversions::to_string_t("");
    m_Open = 0.0;
    m_OpenIsSet = false;
    m_High = 0.0;
    m_HighIsSet = false;
    m_Low = 0.0;
    m_LowIsSet = false;
    m_Close = 0.0;
    m_CloseIsSet = false;
    m_Trades = 0.0;
    m_TradesIsSet = false;
    m_Volume = 0.0;
    m_VolumeIsSet = false;
    m_Vwap = 0.0;
    m_VwapIsSet = false;
    m_LastSize = 0.0;
    m_LastSizeIsSet = false;
    m_Turnover = 0.0;
    m_TurnoverIsSet = false;
    m_HomeNotional = 0.0;
    m_HomeNotionalIsSet = false;
    m_ForeignNotional = 0.0;
    m_ForeignNotionalIsSet = false;
}

TradeBin::~TradeBin()
{
}

void TradeBin::validate()
{
    // TODO: implement validation
}

web::json::value TradeBin::toJson() const
{
    web::json::value val = web::json::value::object();

    val[utility::conversions::to_string_t("timestamp")] = ModelBase::toJson(m_Timestamp);
    val[utility::conversions::to_string_t("symbol")] = ModelBase::toJson(m_Symbol);
    if(m_OpenIsSet)
    {
        val[utility::conversions::to_string_t("open")] = ModelBase::toJson(m_Open);
    }
    if(m_HighIsSet)
    {
        val[utility::conversions::to_string_t("high")] = ModelBase::toJson(m_High);
    }
    if(m_LowIsSet)
    {
        val[utility::conversions::to_string_t("low")] = ModelBase::toJson(m_Low);
    }
    if(m_CloseIsSet)
    {
        val[utility::conversions::to_string_t("close")] = ModelBase::toJson(m_Close);
    }
    if(m_TradesIsSet)
    {
        val[utility::conversions::to_string_t("trades")] = ModelBase::toJson(m_Trades);
    }
    if(m_VolumeIsSet)
    {
        val[utility::conversions::to_string_t("volume")] = ModelBase::toJson(m_Volume);
    }
    if(m_VwapIsSet)
    {
        val[utility::conversions::to_string_t("vwap")] = ModelBase::toJson(m_Vwap);
    }
    if(m_LastSizeIsSet)
    {
        val[utility::conversions::to_string_t("lastSize")] = ModelBase::toJson(m_LastSize);
    }
    if(m_TurnoverIsSet)
    {
        val[utility::conversions::to_string_t("turnover")] = ModelBase::toJson(m_Turnover);
    }
    if(m_HomeNotionalIsSet)
    {
        val[utility::conversions::to_string_t("homeNotional")] = ModelBase::toJson(m_HomeNotional);
    }
    if(m_ForeignNotionalIsSet)
    {
        val[utility::conversions::to_string_t("foreignNotional")] = ModelBase::toJson(m_ForeignNotional);
    }

    return val;
}

void TradeBin::fromJson(web::json::value& val)
{
    setTimestamp
    (ModelBase::dateFromJson(val[utility::conversions::to_string_t("timestamp")]));
    setSymbol(ModelBase::stringFromJson(val[utility::conversions::to_string_t("symbol")]));
    if(val.has_field(utility::conversions::to_string_t("open")))
    {
        web::json::value& fieldValue = val[utility::conversions::to_string_t("open")];
        if(!fieldValue.is_null())
        {
            setOpen(ModelBase::doubleFromJson(fieldValue));
        }
    }
    if(val.has_field(utility::conversions::to_string_t("high")))
    {
        web::json::value& fieldValue = val[utility::conversions::to_string_t("high")];
        if(!fieldValue.is_null())
        {
            setHigh(ModelBase::doubleFromJson(fieldValue));
        }
    }
    if(val.has_field(utility::conversions::to_string_t("low")))
    {
        web::json::value& fieldValue = val[utility::conversions::to_string_t("low")];
        if(!fieldValue.is_null())
        {
            setLow(ModelBase::doubleFromJson(fieldValue));
        }
    }
    if(val.has_field(utility::conversions::to_string_t("close")))
    {
        web::json::value& fieldValue = val[utility::conversions::to_string_t("close")];
        if(!fieldValue.is_null())
        {
            setClose(ModelBase::doubleFromJson(fieldValue));
        }
    }
    if(val.has_field(utility::conversions::to_string_t("trades")))
    {
        web::json::value& fieldValue = val[utility::conversions::to_string_t("trades")];
        if(!fieldValue.is_null())
        {
            setTrades(ModelBase::doubleFromJson(fieldValue));
        }
    }
    if(val.has_field(utility::conversions::to_string_t("volume")))
    {
        web::json::value& fieldValue = val[utility::conversions::to_string_t("volume")];
        if(!fieldValue.is_null())
        {
            setVolume(ModelBase::doubleFromJson(fieldValue));
        }
    }
    if(val.has_field(utility::conversions::to_string_t("vwap")))
    {
        web::json::value& fieldValue = val[utility::conversions::to_string_t("vwap")];
        if(!fieldValue.is_null())
        {
            setVwap(ModelBase::doubleFromJson(fieldValue));
        }
    }
    if(val.has_field(utility::conversions::to_string_t("lastSize")))
    {
        web::json::value& fieldValue = val[utility::conversions::to_string_t("lastSize")];
        if(!fieldValue.is_null())
        {
            setLastSize(ModelBase::doubleFromJson(fieldValue));
        }
    }
    if(val.has_field(utility::conversions::to_string_t("turnover")))
    {
        web::json::value& fieldValue = val[utility::conversions::to_string_t("turnover")];
        if(!fieldValue.is_null())
        {
            setTurnover(ModelBase::doubleFromJson(fieldValue));
        }
    }
    if(val.has_field(utility::conversions::to_string_t("homeNotional")))
    {
        web::json::value& fieldValue = val[utility::conversions::to_string_t("homeNotional")];
        if(!fieldValue.is_null())
        {
            setHomeNotional(ModelBase::doubleFromJson(fieldValue));
        }
    }
    if(val.has_field(utility::conversions::to_string_t("foreignNotional")))
    {
        web::json::value& fieldValue = val[utility::conversions::to_string_t("foreignNotional")];
        if(!fieldValue.is_null())
        {
            setForeignNotional(ModelBase::doubleFromJson(fieldValue));
        }
    }
}

void TradeBin::toMultipart(std::shared_ptr<MultipartFormData> multipart, const utility::string_t& prefix) const
{
    utility::string_t namePrefix = prefix;
    if(namePrefix.size() > 0 && namePrefix.substr(namePrefix.size() - 1) != utility::conversions::to_string_t("."))
    {
        namePrefix += utility::conversions::to_string_t(".");
    }

    multipart->add(ModelBase::toHttpContent(namePrefix + utility::conversions::to_string_t("timestamp"), m_Timestamp));
    multipart->add(ModelBase::toHttpContent(namePrefix + utility::conversions::to_string_t("symbol"), m_Symbol));
    if(m_OpenIsSet)
    {
        multipart->add(ModelBase::toHttpContent(namePrefix + utility::conversions::to_string_t("open"), m_Open));
    }
    if(m_HighIsSet)
    {
        multipart->add(ModelBase::toHttpContent(namePrefix + utility::conversions::to_string_t("high"), m_High));
    }
    if(m_LowIsSet)
    {
        multipart->add(ModelBase::toHttpContent(namePrefix + utility::conversions::to_string_t("low"), m_Low));
    }
    if(m_CloseIsSet)
    {
        multipart->add(ModelBase::toHttpContent(namePrefix + utility::conversions::to_string_t("close"), m_Close));
    }
    if(m_TradesIsSet)
    {
        multipart->add(ModelBase::toHttpContent(namePrefix + utility::conversions::to_string_t("trades"), m_Trades));
    }
    if(m_VolumeIsSet)
    {
        multipart->add(ModelBase::toHttpContent(namePrefix + utility::conversions::to_string_t("volume"), m_Volume));
    }
    if(m_VwapIsSet)
    {
        multipart->add(ModelBase::toHttpContent(namePrefix + utility::conversions::to_string_t("vwap"), m_Vwap));
    }
    if(m_LastSizeIsSet)
    {
        multipart->add(ModelBase::toHttpContent(namePrefix + utility::conversions::to_string_t("lastSize"), m_LastSize));
    }
    if(m_TurnoverIsSet)
    {
        multipart->add(ModelBase::toHttpContent(namePrefix + utility::conversions::to_string_t("turnover"), m_Turnover));
    }
    if(m_HomeNotionalIsSet)
    {
        multipart->add(ModelBase::toHttpContent(namePrefix + utility::conversions::to_string_t("homeNotional"), m_HomeNotional));
    }
    if(m_ForeignNotionalIsSet)
    {
        multipart->add(ModelBase::toHttpContent(namePrefix + utility::conversions::to_string_t("foreignNotional"), m_ForeignNotional));
    }
}

void TradeBin::fromMultiPart(std::shared_ptr<MultipartFormData> multipart, const utility::string_t& prefix)
{
    utility::string_t namePrefix = prefix;
    if(namePrefix.size() > 0 && namePrefix.substr(namePrefix.size() - 1) != utility::conversions::to_string_t("."))
    {
        namePrefix += utility::conversions::to_string_t(".");
    }

    setTimestamp(ModelBase::dateFromHttpContent(multipart->getContent(utility::conversions::to_string_t("timestamp"))));
    setSymbol(ModelBase::stringFromHttpContent(multipart->getContent(utility::conversions::to_string_t("symbol"))));
    if(multipart->hasContent(utility::conversions::to_string_t("open")))
    {
        setOpen(ModelBase::doubleFromHttpContent(multipart->getContent(utility::conversions::to_string_t("open"))));
    }
    if(multipart->hasContent(utility::conversions::to_string_t("high")))
    {
        setHigh(ModelBase::doubleFromHttpContent(multipart->getContent(utility::conversions::to_string_t("high"))));
    }
    if(multipart->hasContent(utility::conversions::to_string_t("low")))
    {
        setLow(ModelBase::doubleFromHttpContent(multipart->getContent(utility::conversions::to_string_t("low"))));
    }
    if(multipart->hasContent(utility::conversions::to_string_t("close")))
    {
        setClose(ModelBase::doubleFromHttpContent(multipart->getContent(utility::conversions::to_string_t("close"))));
    }
    if(multipart->hasContent(utility::conversions::to_string_t("trades")))
    {
        setTrades(ModelBase::doubleFromHttpContent(multipart->getContent(utility::conversions::to_string_t("trades"))));
    }
    if(multipart->hasContent(utility::conversions::to_string_t("volume")))
    {
        setVolume(ModelBase::doubleFromHttpContent(multipart->getContent(utility::conversions::to_string_t("volume"))));
    }
    if(multipart->hasContent(utility::conversions::to_string_t("vwap")))
    {
        setVwap(ModelBase::doubleFromHttpContent(multipart->getContent(utility::conversions::to_string_t("vwap"))));
    }
    if(multipart->hasContent(utility::conversions::to_string_t("lastSize")))
    {
        setLastSize(ModelBase::doubleFromHttpContent(multipart->getContent(utility::conversions::to_string_t("lastSize"))));
    }
    if(multipart->hasContent(utility::conversions::to_string_t("turnover")))
    {
        setTurnover(ModelBase::doubleFromHttpContent(multipart->getContent(utility::conversions::to_string_t("turnover"))));
    }
    if(multipart->hasContent(utility::conversions::to_string_t("homeNotional")))
    {
        setHomeNotional(ModelBase::doubleFromHttpContent(multipart->getContent(utility::conversions::to_string_t("homeNotional"))));
    }
    if(multipart->hasContent(utility::conversions::to_string_t("foreignNotional")))
    {
        setForeignNotional(ModelBase::doubleFromHttpContent(multipart->getContent(utility::conversions::to_string_t("foreignNotional"))));
    }
}

utility::datetime TradeBin::getTimestamp() const
{
    return m_Timestamp;
}


void TradeBin::setTimestamp(utility::datetime value)
{
    m_Timestamp = value;
    
}
utility::string_t TradeBin::getSymbol() const
{
    return m_Symbol;
}


void TradeBin::setSymbol(utility::string_t value)
{
    m_Symbol = value;
    
}
double TradeBin::getOpen() const
{
    return m_Open;
}


void TradeBin::setOpen(double value)
{
    m_Open = value;
    m_OpenIsSet = true;
}
bool TradeBin::openIsSet() const
{
    return m_OpenIsSet;
}

void TradeBin::unsetOpen()
{
    m_OpenIsSet = false;
}

double TradeBin::getHigh() const
{
    return m_High;
}


void TradeBin::setHigh(double value)
{
    m_High = value;
    m_HighIsSet = true;
}
bool TradeBin::highIsSet() const
{
    return m_HighIsSet;
}

void TradeBin::unsetHigh()
{
    m_HighIsSet = false;
}

double TradeBin::getLow() const
{
    return m_Low;
}


void TradeBin::setLow(double value)
{
    m_Low = value;
    m_LowIsSet = true;
}
bool TradeBin::lowIsSet() const
{
    return m_LowIsSet;
}

void TradeBin::unsetLow()
{
    m_LowIsSet = false;
}

double TradeBin::getClose() const
{
    return m_Close;
}


void TradeBin::setClose(double value)
{
    m_Close = value;
    m_CloseIsSet = true;
}
bool TradeBin::closeIsSet() const
{
    return m_CloseIsSet;
}

void TradeBin::unsetClose()
{
    m_CloseIsSet = false;
}

double TradeBin::getTrades() const
{
    return m_Trades;
}


void TradeBin::setTrades(double value)
{
    m_Trades = value;
    m_TradesIsSet = true;
}
bool TradeBin::tradesIsSet() const
{
    return m_TradesIsSet;
}

void TradeBin::unsetTrades()
{
    m_TradesIsSet = false;
}

double TradeBin::getVolume() const
{
    return m_Volume;
}


void TradeBin::setVolume(double value)
{
    m_Volume = value;
    m_VolumeIsSet = true;
}
bool TradeBin::volumeIsSet() const
{
    return m_VolumeIsSet;
}

void TradeBin::unsetVolume()
{
    m_VolumeIsSet = false;
}

double TradeBin::getVwap() const
{
    return m_Vwap;
}


void TradeBin::setVwap(double value)
{
    m_Vwap = value;
    m_VwapIsSet = true;
}
bool TradeBin::vwapIsSet() const
{
    return m_VwapIsSet;
}

void TradeBin::unsetVwap()
{
    m_VwapIsSet = false;
}

double TradeBin::getLastSize() const
{
    return m_LastSize;
}


void TradeBin::setLastSize(double value)
{
    m_LastSize = value;
    m_LastSizeIsSet = true;
}
bool TradeBin::lastSizeIsSet() const
{
    return m_LastSizeIsSet;
}

void TradeBin::unsetLastSize()
{
    m_LastSizeIsSet = false;
}

double TradeBin::getTurnover() const
{
    return m_Turnover;
}


void TradeBin::setTurnover(double value)
{
    m_Turnover = value;
    m_TurnoverIsSet = true;
}
bool TradeBin::turnoverIsSet() const
{
    return m_TurnoverIsSet;
}

void TradeBin::unsetTurnover()
{
    m_TurnoverIsSet = false;
}

double TradeBin::getHomeNotional() const
{
    return m_HomeNotional;
}


void TradeBin::setHomeNotional(double value)
{
    m_HomeNotional = value;
    m_HomeNotionalIsSet = true;
}
bool TradeBin::homeNotionalIsSet() const
{
    return m_HomeNotionalIsSet;
}

void TradeBin::unsetHomeNotional()
{
    m_HomeNotionalIsSet = false;
}

double TradeBin::getForeignNotional() const
{
    return m_ForeignNotional;
}


void TradeBin::setForeignNotional(double value)
{
    m_ForeignNotional = value;
    m_ForeignNotionalIsSet = true;
}
bool TradeBin::foreignNotionalIsSet() const
{
    return m_ForeignNotionalIsSet;
}

void TradeBin::unsetForeignNotional()
{
    m_ForeignNotionalIsSet = false;
}

}
}
}
}

