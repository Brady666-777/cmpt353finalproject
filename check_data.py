import json

vancouver_count = 0
canadian_cities = {}
sample_vancouver = []

print('Searching for Vancouver and Canadian cities...')
with open('data/raw/yelp_academic_dataset_business.json', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i % 10000 == 0:
            print(f'Processed {i} businesses...')
        
        business = json.loads(line)
        city = business.get('city', '').lower()
        state = business.get('state', '')
        
        # Check for Vancouver
        if 'vancouver' in city:
            vancouver_count += 1
            if len(sample_vancouver) < 5:
                sample_vancouver.append({
                    'city': business.get('city'),
                    'state': state,
                    'lat': business.get('latitude'),
                    'lon': business.get('longitude')
                })
        
        # Track Canadian provinces/territories
        canadian_provinces = ['BC', 'AB', 'ON', 'QC', 'MB', 'SK', 'NS', 'NB', 'PE', 'NL', 'NT', 'NU', 'YT']
        if state in canadian_provinces:
            if state not in canadian_cities:
                canadian_cities[state] = 0
            canadian_cities[state] += 1

print(f'\nTotal Vancouver businesses: {vancouver_count}')
print('\nSample Vancouver businesses:')
for biz in sample_vancouver:
    print(f"  {biz['city']}, {biz['state']} - Lat: {biz['lat']}, Lon: {biz['lon']}")

print('\nCanadian businesses by province:')
for prov, count in sorted(canadian_cities.items()):
    print(f'  {prov}: {count}')
