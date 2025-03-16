library(shiny)
library(bslib)

# Define UI for app that draws a histogram ----
ui <- page_sidebar(
  # App title ----
  title = "Variance Gamma Model",
  # Sidebar panel for inputs ----
  sidebar = sidebar(
    # Input: Slider for the value of nu ----
    shinyWidgets::sliderTextInput(
      inputId = "nu",
      label = "nu:",
      choices=c(0.001,0.01,0.1,0.5,0.75,0.99),
      selected=0.5,grid=T
    ),
    # Input: Slider for the value of T ----
    shinyWidgets::sliderTextInput(
      inputId = "T",
      label = "T:",
      choices=c(0.5,1,2,5,10),
      selected=1,grid=T
    )
  ),
  # Output: Distribution ----
  plotOutput(outputId = "distPlot")
)

# Define server logic required to draw a histogram ----
server <- function(input, output) {

  # Probability density function of g ----
  # with requested value of nu
  # This expression that generates a distribution is wrapped in a call
  # to renderPlot to indicate that:
  #
  # 1. It is "reactive" and therefore should be automatically
  #    re-executed when inputs (input$bins) change
  # 2. Its output type is a plot
  ##################################################
  # Function for g

    g_pdf <- function(g,nu,T){
        phi <- dgamma(g,T/nu,scale=nu)
        return(phi)
    }


  output$distPlot <- renderPlot({

    g <- seq(0,15,length.out=500)

    g_prob <- g_pdf(g,input$nu,input$T)

    plot(g,g_prob,type='l',col='blue')

    abline(v=input$T,col='red',lty='dashed')

    title(main="g PDF", xlab="g")

    legend("topright", legend=c("G Distribution", "Maturity (T)"),  
       fill = c("blue","red") 
)

    })

}

shinyApp(ui = ui, server = server)