
<template>
  <div>
    <highcharts :constructor-type="'chart'" :options="chartOptions"></highcharts>
  </div>
</template>

<script>
export default {
  data() {
    return {
      chartOptions: {
        title: {
          text: "Direction"
        },
        series: [
          {
            data: [1, 2, 3, 2, 3, 1]
          },
          {
            data: [1.5, 2.6, 3.7, 2.8, 3.9, 2]
          }
        ]
      }
    };
  }
};
</script>


<style>
.red {
  color: red;
}
</style>
