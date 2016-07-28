library(XML)
library(RCurl)
library(lubridate)

setwd("~/Veronika/USGS")
dir.create("~/Veronika/USGS/weatherSite", showWarnings = TRUE, recursive = FALSE, mode = "0777")
sites <- read.csv("sites.csv")
sites$SITE_NO <- as.factor(sites$SITE_NO)
<<<<<<< HEAD
sites <- sites[sites$SITE_NO != "9535300",]
=======
#sites <- sites[sites$SITE_NO != "9535300",]
>>>>>>> 8d139883cd20b3a0c03faed7777881adef86f2c2

#icao is for city abreviation
#date format = "YYYY/MM/DD"
getNearestLocation <- function(lat, long){
  link <- paste0("http://api.wunderground.com/auto/wui/geo/GeoLookupXML/index.xml?query=", lat,",", long)
  xmldata <- xmlParse(link)
  xmllist <- xmlToList(xmldata)
  #print(paste("The nearest locatuion is", xmllist$nearby_weather_stations$airport$station$city))
  return(xmllist$nearby_weather_stations$airport$station$icao)
}

getweatherYearData <- function(dateStart, dateEnd, icao){
  if(year(dateStart) != year(dateEnd)){
    link <- paste0("https://www.wunderground.com/history/airport/", icao, "/", dateStart,
                   "/CustomHistory.html?dayend=31&monthend=12&yearend=", year(dateStart),
                   "&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&format=1")
  } else {
    link <- paste0("https://www.wunderground.com/history/airport/", icao, "/", dateStart,
                   "/CustomHistory.html?dayend=", day(dateEnd), 
                   "&monthend=", month(dateEnd), 
                   "&yearend=", year(dateEnd), 
                   "&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&format=1")
  }
  data <- read.csv(url(link))
  data <- as.data.frame(cbind(data[,1], data[names(data) %in% c("Max.TemperatureC", "Mean.TemperatureC", "Min.TemperatureC",
                                                                "Max.Humidity", "Mean.Humidity", "Min.Humidity",
                                                                "Mean.Sea.Level.PressurehPa", "Precipitationmm")]))
  data[,1] <- as.Date(data[,1])
  names(data) <- c("Date", "Temperat.Max", "Temperat.Mean", "Temperat.Min", 
                   "Humidity.Max", "Humidity.Mean", "Humidity.Min", "SeaLevel", "Precipitat")
  return(data)
}


getweatherData <- function(dateStart, dateEnd, lat, long){
  icao = getNearestLocation(lat, long)
  totaldata <- data.frame()
  for(i in c(year(dateStart):year(dateEnd))){
    totaldata = as.data.frame(rbind(totaldata, getweatherYearData(dateStart, dateEnd, icao)))
    dateStart = paste(i+1, 1, 1, sep = "/")
  }
  return(totaldata)
}

###test###
###dat = getweatherData("2014/12/01", "2016/06/01", 37.7, -122.36)


errors <- c()
system.time(
<<<<<<< HEAD
  for(j in sites[, "SITE_NO"][450:507]){
=======
  for(j in sites[, "SITE_NO"]){
>>>>>>> 8d139883cd20b3a0c03faed7777881adef86f2c2
    tryCatch({
      lat = sites[which(sites$SITE_NO == j), "DEC_LAT_VA"]
      long = sites[which(sites$SITE_NO == j), "DEC_LONG_V"]
      print(paste(which(sites$SITE_NO == j), j))
      if(nchar(j) != 8){
        j <- paste0(0, j)
      }
      
      fileName <- paste0("/home/vmachine/Veronika/USGS/data/USGS.", j, ".00065.comp.rdb")
      df <- read.delim(file= fileName, header=F, sep = " ", stringsAsFactors = F)
      for(i in 1:nrow(df)){
        if(df[i,1] == "8D	6S	6S	16N	1S	1S	32S	1S"){
          dateStart = paste(substr(df[i+1,1], 1, 4), substr(df[i+1,1], 5, 6), substr(df[i+1,1], 7, 8), sep = "/")
        }
      }
      dateEnd = paste(substr(df[nrow(df),1], 1, 4), substr(df[nrow(df),1], 5, 6), substr(df[nrow(df),1], 7, 8), sep = "/")
      siteWeather = getweatherData(dateStart, dateEnd, lat, long)
      write.csv(siteWeather, paste0("weatherSite/", j, ".csv"))
    },
    
    error = function(e) {
<<<<<<< HEAD
      #print(paste("error with", j));
      errors <- c(errors, j) }
=======
      print(paste("error with", j))}
      #errors <- c(errors, j) }
>>>>>>> 8d139883cd20b3a0c03faed7777881adef86f2c2
    )
    print("done")
  }
)

<<<<<<< HEAD

=======
#sites with no weather info
noweather <- which((!paste0(0,as.character(sites$SITE_NO)) %in% substr(list.files("~/Veronika/USGS/weatherSite"), 1, 8)) &
        (!as.character(sites$SITE_NO) %in% substr(list.files("~/Veronika/USGS/weatherSite"), 1, 8)))
sites[noweather, 1]
>>>>>>> 8d139883cd20b3a0c03faed7777881adef86f2c2
