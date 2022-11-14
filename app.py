import streamlit as st 
import numpy as np
import pandas as pd
import plotly.express as px
import datetime as dt
import streamlit.components.v1 as components
import pycountry
from country_list import available_languages
from country_list import countries_for_language
import time

#####################################################################################

df_sales = pd.read_csv('Data/sales_data_clean.csv', parse_dates=[19,26])
df_inventory = pd.read_csv('Data/Inventory_Stock_Data.csv')

#####################################################################################

# Getting rid of spaces in df_inventory column names

col_rename = {col: col.replace(' ','_') for col in df_inventory.columns}
df_inventory.rename(columns = col_rename, inplace = True)

#####################################################################################

st.set_page_config(
    page_title = 'Supply chain & Inventory Dashboard',
    page_icon = 'alabsai_logo.png',
    layout = 'wide'
)

# dashboard title
col1, col2, col3 = st.columns([1,2,3])

col1.image('alabsai_logo.png')

col2.title("Supply chain & Inventory Dashboard")
col2.markdown('`Submitted by Roderick Anthony`')

col3.markdown('#### **DataCo. Global** is a Logistics company with market all over the world. The supply-chain & Inventory Data is available. The same is to be analysed for insights and recommendation to aid the respective business decisions. The EDA for creating this Dashboard was performed in **`Python`** and the visualizations were created with **`Plotly`**')

#########################################################

# key metrics

#########################################################
total_rev = str(int(df_sales.Sales.sum()))+' $'
units_sold = str(df_sales.Order_Item_Quantity.sum())
current_stock = str(df_inventory.current_stock.sum())

result = pd.merge(df_inventory, df_sales[['Product_Id', 'Class', 'Order_Item_Product_Price']].drop_duplicates(), left_on="product_id", right_on='Product_Id')
result=result.assign(current_stock_value=result.current_stock*result.Order_Item_Product_Price)

current_stock_value = str(int(result.current_stock_value.sum()))+' $'

Gross_profit=str(int(df_sales.Benefit_per_order.sum()))+' $'
Distict_Products=len(df_inventory)
Market_Countries=df_sales.Order_Country.nunique()
Late_deliveries=str((df_sales.Late_Delivery_Risk.value_counts()/len(df_sales)*100).round(2)[0])+' %'


###########################################################

st.header("Sales Level Data")
st.subheader("Here we have Retail Data with the level of granularity at each individual order, spread across a time span of 3 years. Following are a key Insights drawn from the same.")

with st.expander('Across the Globe'):

    map_col, map_select = st.columns([3,1])

    option = map_select.selectbox(
    label='Select the Metric to be indicated by the colour range',
    options=("Profit", "Sales", "Late Deliveries%", "Avg Ship Days", "Total Orders"),
    key="world_map_metric")

    metrics=['Benefit_per_order', 'Sales', 'Shipment_days_real', 'is_late']
    aggs={'Benefit_per_order':'sum', 'Sales':'sum', 'Shipment_days_real':'mean', 'is_late':'mean'}

    world_map_data=df_sales.groupby(['Order_Country', 'Order_cntry_alpha3'])[metrics].agg(aggs).reset_index().round(4)
    # world_map_data=pd.concat(world_map_data, df_sales.Order_Country.value_counts(), axis=1)
    world_map_data=pd.merge(world_map_data,df_sales.Order_Country.value_counts(), left_on='Order_Country', right_index=True)
    world_map_data.drop(columns=['Order_Country_x'], inplace=True)
    world_map_data.Benefit_per_order = world_map_data.Benefit_per_order.astype('int')
    world_map_data.Sales = world_map_data.Sales.astype('int')
    world_map_data.is_late = world_map_data.is_late*100
    world_map_data.Shipment_days_real = world_map_data.Shipment_days_real.round(2)
    rename={'Order_Country':'Country', 'Order_cntry_alpha3':'alpha3', 'Benefit_per_order':'Profit', 
            'Shipment_days_real':'Avg Ship Days', 'is_late':'Late Deliveries%', 'Order_Country_y':'Total Orders'}
    world_map_data.rename(columns=rename, inplace=True)

    fig_world_map = px.choropleth(world_map_data, locations="alpha3",
                        color=st.session_state.world_map_metric, 
                        hover_name="Country", 
                        hover_data=["Profit", "Sales", "Late Deliveries%", "Avg Ship Days", "Total Orders"],
                        color_continuous_scale=px.colors.sequential.Plasma)
    fig_world_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    map_col.plotly_chart(fig_world_map, use_container_width=True)

    map_select.title('{} Countries'.format(Market_Countries))
    
    map_select.metric("Total Revenue", total_rev)
    map_select.metric("Units Sold", units_sold)
    map_select.metric("Gross Profit", Gross_profit)

    st.markdown('#### **The Democratic Republic of the Congo** was represented in the data set by two diferent names: ***República del Congo*** and ***República Democrática del Congo*** along with other irregularities that now stand corrected')


###########################################################

with st.expander('View Key Metrics'):

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", total_rev)
    col2.metric("Units Sold", units_sold)
    col3.metric("Current Stocked Units", current_stock)
    col4.metric("Stock Value", current_stock_value)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gross Profit", Gross_profit)
    col2.metric("Distinct Products", Distict_Products)
    col3.metric("Market Countries", Market_Countries)
    col4.metric("Late Deliveries", Late_deliveries)

##############################################################

# TRENDS

with st.expander('Trends in Total Units Sold, Revenue & Profit across time'):

    df_sales.Order_Date = df_sales.Order_Date.apply(lambda x: x.date())

    df_qnty_trnd=df_sales.groupby('Order_Date')['Order_Item_Quantity'].sum().reset_index().rename(columns={'Order_Date': 'Order Date', 'Order_Item_Quantity': 'Total Units Ordered'})

    fig_qty_trnd = px.line(df_qnty_trnd, x='Order Date', y="Total Units Ordered", title='Units Sold')
    fig_qty_trnd.update_layout(autosize=False, height=350, margin=dict(l=10,r=10,b=10,t=50,pad=4))
    fig_qty_trnd.update_traces(line_color='#8D9EFF')

    st.plotly_chart(fig_qty_trnd, use_container_width=True)

    df_ = df_sales.copy()
    def month_to_num(col):
        return col.apply(lambda x: dt.datetime.strptime(x, '%B').month)
    num_to_month_dict = dict((str(i), dt.datetime.strptime(('0'+str(i))[-2:], '%m').strftime('%B')) for i in range(1,13))
    df_.Order_month = month_to_num(df_.Order_month)

    df_total_orders=pd.pivot_table(df_, values='Order_Item_Quantity', index=['Order_year'], columns=['Order_month', 'Order_week'], aggfunc=np.sum, fill_value=0).round(2)
    col_index=[(num_to_month_dict[str(i)],week) for (i,week) in df_total_orders.columns]
    df_total_orders.columns=pd.MultiIndex.from_tuples(col_index, names=["Order_month", "Order_week"])

    st.dataframe(df_total_orders)

    st.subheader('The above tables gives the Total order quantities by Year, Month and Week, as indicated by the Graph above, the metric sees a Sudden Fall starting from the first week of October 2017')

    df_sales_trnd=df_sales.groupby('Order_Date')['Sales'].sum().reset_index().rename(columns={'Order_Date': 'Order Date', 'Sales': 'Total Sales'})

    fig_sales_trnd = px.line(df_sales_trnd, x='Order Date', y="Total Sales", title='Sales trend')
    fig_sales_trnd.update_layout(autosize=False, height=350, width=1000, margin=dict(l=10,r=10,b=10,t=50,pad=4))
    fig_sales_trnd.update_traces(line_color='#E14D2A')

    st.plotly_chart(fig_sales_trnd, use_container_width=True)

    st.subheader('There are some evident fluctuations towards the end of 2017 & beginning of 2018 worth noting')

    df_profit_trnd=df_sales.groupby('Order_Date')['Benefit_per_order'].sum().reset_index().rename(columns={'Order_Date': 'Order Date', 'Benefit_per_order': 'Profit'})

    fig_profit_trnd = px.line(df_profit_trnd, x='Order Date', y="Profit", title='Profit trend')
    fig_profit_trnd.update_layout(autosize=False, height=350, width=1000, margin=dict(l=10,r=10,b=10,t=50,pad=4))
    fig_profit_trnd.update_traces(line_color='#8EC3B0')

    st.plotly_chart(fig_profit_trnd, use_container_width=True)

################################################################################

with st.expander('Order and Delivery Status'):

    df_delivery_status=df_sales.Delivery_Status.value_counts().reset_index().rename(columns={'index': 'Delivery Status', 'Delivery_Status': 'Total Orders'})
    fig_delivery_status = px.pie(df_delivery_status, values='Total Orders', names='Delivery Status', title='Delivery Status', hole=.3)
    fig_delivery_status.update_layout(autosize=False, height=360, width=360, margin=dict(l=10,r=10,b=10,t=50,pad=4))
    fig_delivery_status.update_layout(showlegend=False)

    df_ordr_status=df_sales.Order_Status.value_counts().reset_index().rename(columns={'index': 'Order Status', 'Order_Status': 'Total Orders'})
    fig_ordr_status = px.bar(df_ordr_status, x="Total Orders", y="Order Status", orientation='h', title='Order Status')
    fig_ordr_status.update_layout(autosize=False, height=360, margin=dict(l=10,r=10,b=10,t=50,pad=4))

    df_late_delivery=df_sales.Late_Delivery_Risk.value_counts().reset_index().rename(columns={'index': 'Late Delivery Risk', 'Late_Delivery_Risk': 'Total Orders'})
    fig_late_delivery = px.pie(df_late_delivery, values='Total Orders', names='Late Delivery Risk', title='Late Delivery Risk', hole=.3)
    fig_late_delivery.update_layout(autosize=False, height=360, width=360, margin=dict(l=10,r=10,b=10,t=50,pad=4))
    fig_late_delivery.update_layout(showlegend=False)

    col1, col2, col3 = st.columns([3,6,3])

    col1.plotly_chart(fig_delivery_status, use_container_width=True)
    col2.plotly_chart(fig_ordr_status, use_container_width=True)
    col3.plotly_chart(fig_late_delivery, use_container_width=True)

########################################################################################

with st.expander('Inventory & Sales by Product Cluster/Class'):

    result = pd.merge(df_inventory, df_sales[['Product_Id', 'Class', 'Order_Item_Product_Price']].drop_duplicates(), left_on="product_id", right_on='Product_Id')
    result=result.assign(current_stock_value=result.current_stock*result.Order_Item_Product_Price)
    df1=result.groupby('Class')[['current_stock', 'current_stock_value']].sum()
    df2=df_sales.groupby('Class')[['Sales', 'Order_Item_Quantity', 'Benefit_per_order']].sum()

    class_color_map={'Small Value-Large Number':'#FF577F',
                    'Moderate Value-Moderate Number':'#6C4AB6',
                    'High Value-Small Number':'#82CD47'}

    df_class=pd.concat([df1, df2], axis=1).reset_index().rename(columns={'Order_Item_Quantity': 'Orders', 'Benefit_per_order': 'Profit', 'current_stock_value': 'Stock Value', 'current_stock': 'Stock'}).round()

    fig_class_stock = px.pie(df_class, values='Stock', names='Class', title='Stock by class', hole=.3, color='Class', color_discrete_map=class_color_map)
    fig_class_stock.update_layout(width=300, height=300, margin=dict(l=0,r=10,b=10,t=50, pad=4), showlegend=False)
    
    Small_Value_Large_Number=df_class[df_class.Class=='Small Value-Large Number']['Sales'].sum()
    Moderate_Value_Moderate_Number=df_class[df_class.Class=='Moderate Value-Moderate Number']['Sales'].sum()
    High_Value_Small_Number=df_class[df_class.Class=='High Value-Small Number']['Sales'].sum()

    col1, col2, col3 = st.columns(3)

    col1.plotly_chart(fig_class_stock, use_container_width=True)

    fig_class_sales = px.pie(df_class, values='Sales', names='Class', title='Sales by class', hole=.3, color='Class', color_discrete_map=class_color_map)
    fig_class_sales.update_layout(width=300, height=300, margin=dict(l=0,r=10,b=10,t=50, pad=4), showlegend=False)

    col2.plotly_chart(fig_class_sales)

    col3.markdown(
        """
        ### "High Value-Small Number" dominates Sales at 52%, while it's average stock over the given span sits at under 4%. The disproportionate Stock-Sales pairs need to be regulated and Top-Selling products need to be prioritised in the inventory

        """
    )
    # col2.metric("Small Value-Large Number", int(Small_Value_Large_Number))
    # col2.metric("Moderate Value-Moderate Number", int(Moderate_Value_Moderate_Number))
    # col2.metric("High Value-Small Number", int(High_Value_Small_Number))

###############################################################################

with st.expander('Metrics by Year'):

    metrics=['Benefit_per_order', 'Sales', 'is_late']
    aggs={'Benefit_per_order':'sum', 'Sales':'sum', 'is_late':'mean'}

    df_yearly=df_sales.groupby('Order_year')[metrics].agg(aggs).reset_index()
    # world_map_data=pd.concat(world_map_data, df_sales.Order_Country.value_counts(), axis=1)
    df_yearly=pd.merge(df_yearly,df_sales.Order_year.value_counts(), left_on='Order_year', right_index=True)
    df_yearly.drop(columns=['Order_year_x'], inplace=True)
    df_yearly.Benefit_per_order = df_yearly.Benefit_per_order.astype('int')
    df_yearly.Sales = df_yearly.Sales.astype('int')
    df_yearly.is_late = df_yearly.is_late*100
    rename={'Order_year':'Year', 'Benefit_per_order':'Profit', 
            'is_late':'Late Deliveries%', 'Order_year_y':'Total Orders'}
    df_yearly.rename(columns=rename, inplace=True)

    fig_yearly = px.bar(df_yearly, x="Total Orders", y="Year", orientation='h',
                hover_data=['Sales', 'Profit', 'Late Deliveries%'], height=400)
    fig_yearly.update_layout(autosize=False, height=350, width=1000, margin=dict(l=10,r=10,b=10,t=10,pad=4))

    col1, col2 = st.columns([3,1])

    col1.plotly_chart(fig_yearly, use_container_width=True)

    col2.subheader('A lower sum of Total Orders in 2017, indicates towords the steep decline in that metric starting from October 2017')

###############################################################################

with st.expander('Top 10 Countries inTotal Orders'):

    units_by_country=df_sales.groupby('Order_Country')[['Order_Item_Quantity', 'Sales', 'Benefit_per_order']].sum().sort_values(by='Order_Item_Quantity', ascending=False)
    cntry_ordrs=df_sales.Order_Country.value_counts()
    cntry_late=df_sales.groupby('Order_Country')['is_late'].mean()*100
    cntry_df=pd.concat([units_by_country, cntry_ordrs, cntry_late], axis=1).reset_index().rename(columns={'index': 'Country', 'Order_Country': 'Total Orders', 'Benefit_per_order': 'Profit', 'is_late': 'Late Delivery%'})
    cntry_df=cntry_df.sort_values(by='Sales', ascending=False).head(10).round(2)

    fig_cntry = px.bar(cntry_df, x="Total Orders", y="Country", orientation='h',
                hover_data=['Sales', 'Profit', 'Late Delivery%'], color='Late Delivery%', height=400)
    fig_cntry.update_layout(autosize=False, height=350, width=1000, margin=dict(l=10,r=10,b=10,t=10,pad=4))

    col1, col2 = st.columns([3,1])

    col1.plotly_chart(fig_cntry, use_container_width=True)

    col2.subheader('The percentage of Late Deliveries seems to lie in a narrow range')
    col2.subheader('Moreover the top 3 countries exceed the rest by a significant mark')

###############################################################################

with st.expander('Relationship between Shipping mode & Late Delivery'):
    col1, col2 = st.columns([2,2])
    df_ship=pd.crosstab(df_sales.Shipping_Mode, df_sales.Late_Delivery_Risk, margins=True)

    df_ship_data=pd.concat([df_sales.Shipping_Mode.value_counts(),df_sales.groupby('Shipping_Mode')['is_late'].mean()*100,df_sales[df_sales.is_late].Shipping_Mode.value_counts()],axis=1)
    df_ship_data=df_ship_data.reset_index()
    df_ship_data.columns=['Shipping Mode' ,'Total Orders', 'Late Delivery%', 'Total Late Deliveries']
    fig_ship = px.pie(df_ship_data, values='Total Late Deliveries', names='Shipping Mode', hover_data=['Late Delivery%', 'Total Orders'], hole=.3, title='Total Late Deliveries')
    fig_ship.update_layout(margin=dict(l=0,r=10,b=10,t=50,pad=4), showlegend=False)

    col1.plotly_chart(fig_ship, use_container_width=True)
    col2.dataframe(data=df_ship, use_container_width=True)
    col2.markdown('# Standard Class')
    col2.subheader('Proves to be the most promising mode. Although evident by a visual inspection, the chi2_contingency test verifies that Late Deliveries are Dependent on the Shipping mode.')
    
###############################################################################
 
st.header("Inventory Level Data")
st.subheader("Unlike the Sales data that provides a historical account of order details at the granularity of each order itself, the Inventory data is a snapshot of the inventory at a given point with some additional information to our aid.")

df_inventory.rename(columns={'order-now':'Order_now'}, inplace=True)

result = pd.merge(df_inventory, df_sales[['Product_Id', 'Category_Name', 'Class', 'Order_Item_Product_Price']].drop_duplicates(), left_on="product_id", right_on='Product_Id')
result.drop(columns=['product_id'], inplace=True)

agg_prod={'Order_Item_Quantity': 'sum', 'Sales': 'sum', 'Benefit_per_order': 'sum', 'is_late': 'mean'}
df_prod_id=df_sales.groupby('Product_Id')[['Order_Item_Quantity', 'Sales', 'Benefit_per_order', 'is_late']].agg(agg_prod)
df_prod_id.columns=['Units_Sold', 'Sales', 'Profit', 'Late_Delivery%']
df_prod_id['Late_Delivery%']=df_prod_id['Late_Delivery%']*100
df_prod_id = df_prod_id.round(2)
df_prod_id = pd.merge(result, df_prod_id, left_on="Product_Id", right_index=True)
df_prod_id = df_prod_id.assign(sells_per_reorder=df_prod_id.Units_Sold/df_prod_id.reorder_point)
df_prod_id = df_prod_id.assign(understocked=df_prod_id.current_stock==df_prod_id.reorder_point)
df_prod_id.Sales = df_prod_id.Sales.astype('int')
df_prod_id.sells_per_reorder = df_prod_id.sells_per_reorder.astype('int')
df_prod_id.Profit = df_prod_id.Profit.astype('int')

with st.expander('Top 10 Products by Unis sold'):
    col1, col2 = st.columns([5,2])

    fig_prod = px.bar(df_prod_id.nlargest(10, 'Units_Sold'), x="Units_Sold", y="product_name", orientation='h', hover_data=['Sales', 'Profit',  'sells_per_reorder'], color='understocked', height=400)
    fig_prod.update_layout(autosize=False, height=350, width=1000, margin=dict(l=10,r=10,b=10,t=10,pad=4))

    col1.plotly_chart(fig_prod, use_container_width=True)
    col2.markdown("##### It's reasonable to assign a reorder point to products that is proportional to Units Sold, in order to maintain a consistency in the supply chain flow. Since, the Average & Maximum lead time is constant for all products, including the Manufacturing on demand products, the only factor dictating reorders must be Demand.")
    col2.subheader("3 Products in the Top 10 by Units Sold are understocked")

###############################################################################

with st.expander('Top 10 Products by Profit'):
    col1, col2 = st.columns([5,2])

    fig_prod = px.bar(df_prod_id.nlargest(10, 'Profit'), x="Profit", y="product_name", orientation='h', hover_data=['Sales', 'Profit',  'sells_per_reorder'], color='understocked', height=400)
    fig_prod.update_layout(autosize=False, height=350, width=1000, margin=dict(l=10,r=10,b=10,t=10,pad=4))

    col1.plotly_chart(fig_prod, use_container_width=True)
    col2.header("3 among the Top 10 most profit generating Products are understocked")

###############################################################################

with st.expander('Bottom 10 Products by Profit'):
    col1, col2 = st.columns([5,2])

    fig_prod = px.bar(df_prod_id.nsmallest(10, 'Profit'), x="Profit", y="product_name", orientation='h', hover_data=['Sales', 'Profit',  'sells_per_reorder'], color='Sales', height=400)
    fig_prod.update_layout(autosize=False, height=350, width=1000, margin=dict(l=10,r=10,b=10,t=10,pad=4))

    col1.plotly_chart(fig_prod, use_container_width=True)
    col2.header("Out of the 3 products running in loss, the one in the most loss has staggeringly high Sales.")
    col2.markdown('This should raise some alarms. Repricing thse products or discontinuing them should be considered.')

###############################################################################

top_10_stock=df_prod_id[['Product_Id', 'product_name', 'current_stock', 'reorder_point', 'Units_Sold', 'Sales', 'sells_per_reorder']].nlargest(10, 'current_stock')

top_10_sells=df_prod_id[['Product_Id', 'product_name', 'current_stock', 'reorder_point', 'Units_Sold', 'Sales', 'sells_per_reorder']].nlargest(10, 'Units_Sold')

stock_nd_sells = set(top_10_stock.Product_Id.values).intersection(set(top_10_sells.Product_Id.values))

def highlight_rows(x):
    if x.Product_Id in (stock_nd_sells):
        return['background-color: #FF9F9F']*7
    else:
        return['background-color: #F7F7F7']*7
    
top_10_stock=top_10_stock.style.apply(highlight_rows, axis = 1)

top_10_sells=top_10_sells.style.apply(highlight_rows, axis = 1)

with st.expander('Overstocked Products'):
    col1, col2 = st.columns(2)

    col1.markdown('Top 10 most Stocked Products')
    col1.write(top_10_stock)

    col2.markdown('Top 10 most Selling Products')
    col2.write(top_10_sells)

    col1, col2 = st.columns(2)

    col2.slider('Select Minimimum Total Units Sold', min_value=0, max_value=6000, step=500, key="min_units_sold")
    col2.slider('Select Minimimum Total Sales', min_value=1000, max_value=500000, step=50000, key="min_sales")

    s=st.session_state["min_sales"]
    u=st.session_state["min_units_sold"]

    qry='Sales > {} & Units_Sold > {}'.format(s,u)

    overstocked=df_prod_id[['Product_Id', 'product_name', 'current_stock', 'reorder_point', 'Units_Sold', 'Sales', 'sells_per_reorder']].query(qry).nsmallest(10,'sells_per_reorder')

    col1.markdown('Top 10 Overstocked Products by the Adjacent Filters')
    col1.write(overstocked)


    col2.markdown(
        """
        ##### There's an overlap of just one product in Top Products by Stock & Units sold respectively. Moreover the Units sold per reorder point is exceedingly low for the rest of those products, hence the reorder point needs to be recalibrated for such products.

        ### Units sold per Reorder Point
        ##### Combined with the desired range of filters can fetch us some insights to reconsider the Inventory management.

        """)

##########################################################################
    
with st.expander('Understocked Products'):

    col1, col2 = st.columns(2)

    col2.slider('Select Minimimum Total Units Sold', min_value=0, max_value=6000, step=500, key="min_units_sold_2")
    col2.slider('Select Minimimum Total Sales', min_value=1000, max_value=500000, step=50000, key="min_sales_2")

    s=st.session_state["min_sales_2"]
    u=st.session_state["min_units_sold_2"]

    qry='Sales > {} & Units_Sold > {}'.format(s,u)

    overstocked=df_prod_id[['Product_Id', 'product_name', 'current_stock', 'reorder_point', 'Units_Sold', 'Sales', 'sells_per_reorder']].query(qry).nlargest(10,'sells_per_reorder')

    col1.markdown('Top 10 Understocked Products by the Adjacent Filters')
    col1.write(overstocked)

    col2.markdown(
        """
        ### Units sold per Reorder Point in the adjoining table is exceedingly high.
        ##### Combined with the desired range of filters we can fetch some insights to reconsider the Inventory management.

        """)

    col2.caption('As with any Dynamic Data Set, with more time & Domain knowledge, many more insights can be drawn. However, this is all for now. Thankyou')
    col2.markdown('#### `Submitted by Roderick Anthony`')
