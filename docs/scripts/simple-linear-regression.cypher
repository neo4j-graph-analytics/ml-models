// tag::initialize[]

CALL regression.linear.create('airbnb prices')

// end::initialize[]

// tag::add[]

MATCH (list:Listing)-[:IN_NEIGHBORHOOD]->(:Neighborhood{neighborhood_id:'78752'})
WHERE exists(list.bedrooms)
  AND exists(list.price)
  AND NOT exists(list.added) OR list.added = false
CALL regression.linear.add('airbnb prices', list.bedrooms, list.price)
SET list.added = true
RETURN list.listing_id

// end::add[]

// tag::addM[]

MATCH (list:Listing)-[:IN_NEIGHBORHOOD]->(:Neighborhood{neighborhood_id:'78752'})
WHERE exists(list.bedrooms)
  AND exists(list.price)
  AND NOT exists(list.added) OR list.added = false
SET list.added = true
WITH collect(list.bedrooms) AS bedrooms, collect(list.price) AS prices
CALL regression.linear.addM('airbnb prices', bedrooms, prices)
RETURN bedrooms, prices

// end::addM[]

// tag::predict[]

RETURN regression.linear.predict('airbnb prices', 4)

// end::predict[]

// tag::predict-and-store[]

MATCH (list:Listing)-[:IN_NEIGHBORHOOD]->(:Neighborhood {neighborhood_id:’78752’})
WHERE exists(list.bedrooms) AND NOT exists(list.price)
SET list.predicted_price = regression.linear.predict(list.bedrooms)

// end::predict-and-store[]

// tag::remove[]

MATCH (list:Listing {listing_id:2467149})-[:IN_NEIGHBORHOOD]->(:Neighborhood {neighborhood_id:’78752’})
CALL regression.linear.remove(‘airbnb prices’, list.bedrooms, list.price)
SET list.added = false

// end::remove[]

// tag::add-more[]

MATCH (list:Listing)-[:IN_NEIGHBORHOOD]->(:Neighborhood {neighborhood_id:’78753’})
WHERE exists(list.bedrooms)
AND exists(list.price)
AND NOT exists(list.added) OR list.added = false
CALL regression.linear.add(‘airbnb prices’, list.bedrooms, list.price) RETURN list

// end::add-more[]

// tag::info[]

CALL regression.linear.info(‘airbnb prices’)
YIELD model, state, N
RETURN model, state, N

// end::info[]

// tag::stats[]

CALL regression.linear.stats(‘airbnb prices’)
YIELD intercept, slope, rSquare, significance
RETURN intercept, slope, rSquare, significance

// end::stats[]

// tag::serialize[]

MERGE (m:ModelNode{model: ‘airbnb prices’})
SET m.data = regression.linear.serialize(‘airbnb prices’)
RETURN m

// end::serialize[]

// tag::delete[]

CALL regression.linear.delete(‘airbnb prices’)
YIELD model, state, N
RETURN model, state, N

// end::delete[]

// tag::load[]

MATCH (m:ModelNode{model: ‘airbnb prices’})
CALL regression.linear.load(‘airbnb prices’, m.data)

// end::load[] 
