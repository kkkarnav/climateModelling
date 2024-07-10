#!/bin/bash
declare -a cities=(
"Mumbai 19.0760 72.8777"
"Delhi 28.7041 77.1025"
"Bangalore 12.9716 77.5946"
"Hyderabad 17.3850 78.4867"
"Ahmedabad 23.0225 72.5714"
"Chennai 13.0827 80.2707"
"Kolkata 22.5726 88.3639"
"Surat 21.1702 72.8311"
"Pune 18.5204 73.8567"
"Jaipur 26.9124 75.7873"
"Lucknow 26.8467 80.9462"
"Kanpur 26.4499 80.3319"
"Nagpur 21.1458 79.0882"
"Indore 22.7196 75.8577"
"Bhopal 23.2599 77.4126"
"Visakhapatnam 17.6868 83.2185"
"Patna 25.5941 85.1376"
"Vadodara 22.3072 73.1812"
"Ludhiana 30.9009 75.8573"
"Agra 27.1767 78.0081"
"Nashik 19.9975 73.7898"
"Faridabad 28.4089 77.3178"
"Meerut 28.9845 77.7064"
"Rajkot 22.3039 70.8022"
"Varanasi 25.3176 82.9739"
"Srinagar 34.0837 74.7973"
"Aurangabad 19.8762 75.3433"
"Dhanbad 23.7957 86.4304"
"Amritsar 31.6340 74.8723"
"Allahabad 25.4358 81.8463"
"Ranchi 23.3441 85.3096"
"Gwalior 26.2183 78.1828"
"Jabalpur 23.1815 79.9864"
)

for city_info in "${cities[@]}"
do
    name=$(echo $city_info | awk '{print $1}')
    lat=$(echo $city_info | awk '{print $2}')
    lon=$(echo $city_info | awk '{print $3}')
    python grid_wise_plots.py --name "$name" --latitude $lat --longitude $lon
done
