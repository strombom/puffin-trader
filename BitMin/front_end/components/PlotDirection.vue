
<template>
  <div>
    <highcharts :constructor-type="'chart'" :options="chartOptions"></highcharts>
    <div>
      <button type="button" class="btn btn-primary" @click="updateChart">Refresh</button>
    </div>
  </div>
</template>

<script>
import { ChartEventBus } from "./../plugins/chart_event_bus.js";
import Highcharts from "highcharts";
//import axios from 'axios'

//axios.defaults.baseURL = process.env.API_URL
//console.log("axios.defaults.baseURL " + $axios.defaults.baseURL)

let exchanges = {
  bitmex: {
    index: 0,
    name: "BitMex"
  },
  binance: {
    index: 1,
    name: "Binance"
  }
};

export default {
  data() {
    return {
      chartOptions: {
        title: {
          text: "Direction"
        },
        chart: {
          type: 'line'
        },
        series: [
          {
            data: [1, 2, 3, 2, 3, 1],
            name: exchanges['bitmex']['name']
          },
          {
            data: [1.5, 2.6, 3.7, 2.8, 3.9, 2],
            name: exchanges['binance']['name']
          }
        ]
      }
    };
  },
  methods: {
    updateChart: async function(event) {
      let new_data
      //new_data = requestData()


      const data = (await this.$axios.get("/directions")).data

      console.log(data.prices.bitmex)

      Highcharts.charts.forEach((chart, chart_index) => {
        Object.entries(exchanges).forEach(([exchange_key, exchange]) => {
            let series_idx = exchange["index"]
            chart.series[series_idx].setData(data.prices[exchange_key], true)
        });
      });
    }
  }
};




async function requestData() {
  //const data = await this.$axios.get("/directions")
  

  console.log("abc")
  return {
    bitmex: [5.5, 1.6, 2.7, 1.8, 2.9, 1],
    binance: [0, 2, 1, 2, 1, 2]
  }
}
</script>

<style>
.red {
  color: red;
}
</style>
