namespace Neural_Network
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.lblTitle = new System.Windows.Forms.Label();
            this.lblDescription = new System.Windows.Forms.Label();
            this.btnTrain = new System.Windows.Forms.Button();
            this.lblResults = new System.Windows.Forms.Label();
            this.txtResults = new System.Windows.Forms.RichTextBox();
            this.SuspendLayout();
            // 
            // lblTitle
            // 
            this.lblTitle.AutoSize = true;
            this.lblTitle.Location = new System.Drawing.Point(29, 29);
            this.lblTitle.Name = "lblTitle";
            this.lblTitle.Size = new System.Drawing.Size(122, 13);
            this.lblTitle.TabIndex = 0;
            this.lblTitle.Text = "Neural Network Training";
            // 
            // lblDescription
            // 
            this.lblDescription.AutoSize = true;
            this.lblDescription.Location = new System.Drawing.Point(29, 51);
            this.lblDescription.Name = "lblDescription";
            this.lblDescription.Size = new System.Drawing.Size(227, 13);
            this.lblDescription.TabIndex = 1;
            this.lblDescription.Text = "4 inputs • 5 hidden neurons • 3 output classes";
            // 
            // btnTrain
            // 
            this.btnTrain.Location = new System.Drawing.Point(32, 88);
            this.btnTrain.Name = "btnTrain";
            this.btnTrain.Size = new System.Drawing.Size(119, 41);
            this.btnTrain.TabIndex = 2;
            this.btnTrain.Text = "Train Network";
            this.btnTrain.UseVisualStyleBackColor = true;
            this.btnTrain.Click += new System.EventHandler(this.btnTrain_Click);
            // 
            // lblResults
            // 
            this.lblResults.AutoSize = true;
            this.lblResults.Location = new System.Drawing.Point(29, 177);
            this.lblResults.Name = "lblResults";
            this.lblResults.Size = new System.Drawing.Size(83, 13);
            this.lblResults.TabIndex = 3;
            this.lblResults.Text = "Training Results";
            // 
            // txtResults
            // 
            this.txtResults.Font = new System.Drawing.Font("Consolas", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtResults.Location = new System.Drawing.Point(32, 193);
            this.txtResults.Name = "txtResults";
            this.txtResults.Size = new System.Drawing.Size(314, 219);
            this.txtResults.TabIndex = 4;
            this.txtResults.Text = "";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(379, 481);
            this.Controls.Add(this.txtResults);
            this.Controls.Add(this.lblResults);
            this.Controls.Add(this.btnTrain);
            this.Controls.Add(this.lblDescription);
            this.Controls.Add(this.lblTitle);
            this.Name = "Form1";
            this.Text = "Form1";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label lblTitle;
        private System.Windows.Forms.Label lblDescription;
        private System.Windows.Forms.Button btnTrain;
        private System.Windows.Forms.Label lblResults;
        private System.Windows.Forms.RichTextBox txtResults;
    }
}

